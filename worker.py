#!/usr/bin/env python3
"""Receive moderation tasks and return one publication decision per comment."""
import argparse
import asyncio
from dataclasses import dataclass, field
import fcntl
import json
import logging
import math
import os
import re
import signal
import socket
import time
from urllib.parse import urlsplit

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from model import FatalInference, LABELS, Model, check_runtime
from policy import CATEGORIES, LivePolicy, ROOT, validate_text

LOG = logging.getLogger("clef2")
SEND_TIMEOUT = 5


def log_event(event, **fields):
    LOG.info(json.dumps({"event": event, **fields}, ensure_ascii=True, allow_nan=False))


@dataclass(frozen=True)
class Settings:
    url: str
    protocol: str = field(repr=False)
    timeout: float = 30
    capacity: int = 16

    @classmethod
    def from_env(cls):
        url, token = os.environ["MODERATION_URL"], os.environ["MODERATION_TOKEN"]
        parsed = urlsplit(url)
        if (parsed.scheme not in {"ws", "wss"} or not parsed.hostname
                or parsed.path != "/ws/moderation" or parsed.query or parsed.fragment
                or parsed.username or parsed.password):
            raise ValueError("invalid_moderation_url")
        parsed.port
        if parsed.scheme == "ws" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("wss_required")
        if not re.fullmatch(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+", token):
            raise ValueError("invalid_moderation_token")
        timeout = float(os.environ.get("CLEF_INFERENCE_TIMEOUT_SECONDS", "30"))
        capacity = int(os.environ.get("CLEF_QUEUE_CAPACITY", "16"))
        if not math.isfinite(timeout) or timeout <= 0 or capacity <= 0:
            raise ValueError("invalid_worker_limits")
        return cls(url, "bearer." + token, timeout, capacity)


def task_id(task):
    if not isinstance(task, dict) or task.get("type") != "moderation.task":
        return None
    value = task.get("comment_id")
    if not isinstance(value, str) or not value:
        return None
    try:
        value.encode("utf-8")
    except UnicodeError:
        return None
    return value


def result_message(comment_id, result):
    if (not isinstance(result, dict) or not isinstance(result.get("decision"), str)
            or result["decision"] not in LABELS):
        raise ValueError("invalid_model_decision")
    categories = result.get("categories")
    if (not isinstance(categories, list) or any(not isinstance(c, str) or c not in CATEGORIES for c in categories)
            or len(set(categories)) != len(categories)
            or (result["decision"] != "block" and categories)):
        raise ValueError("invalid_categories")
    return {"type": "moderation.result", "comment_id": comment_id, "worker_id": "clef",
            "decision": result["decision"], "categories": categories}


def review_message(comment_id):
    return result_message(comment_id, {"decision": "review", "categories": []})


def reject_constant(value):
    raise ValueError("invalid_json_constant")


async def session(ws, policy, judge, settings):
    queue = asyncio.Queue(maxsize=settings.capacity)
    send_lock = asyncio.Lock()
    current = None
    sending_current = False
    inference = None

    async def send(message):
        async with asyncio.timeout(SEND_TIMEOUT):
            async with send_lock:
                await ws.send(json.dumps(message, ensure_ascii=True, allow_nan=False))
        log_event("result", comment_id=message["comment_id"], decision=message["decision"])

    async def receive():
        async for raw in ws:
            try:
                task = json.loads(raw, parse_constant=reject_constant)
            except (ValueError, UnicodeError):
                log_event("error", error="invalid_json")
                continue
            if not isinstance(task, dict):
                log_event("error", error="invalid_message")
                continue
            if task.get("type") != "moderation.task":
                if task.get("type") == "error":
                    # Log only a known code; server error text may contain user data.
                    code = task.get("code")
                    log_event("server_error", error=code if code in ("already_decided", "not_found", "forbidden")
                              else "other")
                continue
            comment_id = task_id(task)
            if comment_id is None:
                log_event("error", error="invalid_comment_id")
                continue
            try:
                if task.get("channel_id") not in ("uni", "1A", "kaikan"):
                    raise ValueError("invalid_channel")
                validate_text(task.get("text"))
            except ValueError:
                log_event("error", comment_id=comment_id, error="invalid_input")
                await send(review_message(comment_id))
                continue
            try:
                queue.put_nowait(task)
            except asyncio.QueueFull:
                log_event("error", comment_id=comment_id, error="queue_full")
                await send(review_message(comment_id))

    async def process():
        nonlocal current, sending_current, inference
        while True:
            current = await queue.get()
            sending_current = False
            started = time.monotonic()
            comment_id = current["comment_id"]
            try:
                result = policy.static_result(current["text"])
                if result is None:
                    inference = asyncio.create_task(judge(current["text"]))
                    done, _ = await asyncio.wait({inference}, timeout=settings.timeout)
                    if not done:
                        inference.cancel()
                        raise FatalInference("inference_timeout")
                    result = inference.result()
                    # Only static rules can supply categories.
                    if result.get("categories") != []:
                        raise ValueError("invalid_model_categories")
                message = result_message(comment_id, result)
            except FatalInference as exc:
                log_event("error", comment_id=comment_id, error=str(exc))
                raise
            except Exception as exc:
                log_event("error", comment_id=comment_id, error="judgement_failed", kind=type(exc).__name__)
                message = review_message(comment_id)
            log_event("judged", comment_id=comment_id, elapsed_ms=round((time.monotonic() - started) * 1000),
                      decision=message["decision"])
            sending_current = True
            await send(message)
            current = None
            inference = None

    receiver = asyncio.create_task(receive())
    processor = asyncio.create_task(process())
    try:
        done, _ = await asyncio.wait({receiver, processor}, return_when=asyncio.FIRST_COMPLETED)
        # Prefer a fatal engine failure if both tasks finished together.
        if processor in done:
            processor.result()
        if receiver in done:
            receiver.result()
    finally:
        tasks = [receiver, processor]
        if inference is not None:
            tasks.append(inference)
        for task in tasks:
            if not task.done():
                task.cancel()
        done, pending = await asyncio.wait(tasks, timeout=SEND_TIMEOUT)
        for task in done:
            if not task.cancelled():
                task.exception()
        remaining = []
        if current is not None and not sending_current:
            remaining.append(current)
        while not queue.empty():
            remaining.append(queue.get_nowait())
        try:
            async with asyncio.timeout(SEND_TIMEOUT):
                for task in remaining:
                    await send(review_message(task["comment_id"]))
        except (ConnectionClosed, OSError, TimeoutError):
            log_event("error", error="shutdown_review_failed")
        if pending:
            raise FatalInference("inference_cancel_failed")


async def run(settings):
    policy = LivePolicy.load()
    model = Model(policy)
    loop = asyncio.get_running_loop()
    running = asyncio.current_task()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, running.cancel)
    # websockets debug logs can include handshake credentials.
    websocket_log = logging.getLogger("clef2.websocket")
    websocket_log.disabled = True
    try:
        async for ws in connect(settings.url, subprotocols=[settings.protocol], max_size=65536,
                                max_queue=16, compression=None, proxy=None, close_timeout=3,
                                logger=websocket_log):
            log_event("connected")
            try:
                await session(ws, policy, model.decide, settings)
            except (ConnectionClosed, OSError, TimeoutError):
                log_event("disconnected")
            await asyncio.sleep(3)
    finally:
        model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify runtime without starting GPU or WebSocket")
    parser.add_argument("--verify-model", action="store_true", help="also hash the large model weights")
    args = parser.parse_args()
    try:
        check_runtime(verify_model=args.verify_model)
        if args.check:
            log_event("runtime_verified")
            return 0
        settings = Settings.from_env()
        with (ROOT / "runtime/worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # The existing API uses this GPU; do not allocate a second engine beside it.
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", 11634)) == 0:
                    raise RuntimeError("existing_gpu_service_running")
            asyncio.run(run(settings))
        return 0
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 0
    except Exception as exc:
        log_event("fatal", error=type(exc).__name__)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
