import asyncio
import json
import math
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import InvalidStatus

from model import FatalInference
from policy import LivePolicy, ROOT
import worker

TASK = {"type": "moderation.task", "comment_id": "first", "channel_id": "uni", "text": "がんばれ"}


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.attempts = []
        self.fail_send = False

    def __aiter__(self): return self

    async def __anext__(self):
        raw = await self.incoming.get()
        if raw is None:
            raise StopAsyncIteration
        return raw

    async def send(self, raw):
        self.attempts.append(json.loads(raw))
        if self.fail_send:
            raise OSError("closed")
        await self.outgoing.put(json.loads(raw))

    async def push(self, task):
        await self.incoming.put(json.dumps(task))

    async def result(self):
        return await asyncio.wait_for(self.outgoing.get(), 2)


class SettingsTests(unittest.TestCase):
    def test_defaults_auth_and_limits(self):
        env = {"MODERATION_URL": "wss://example.org/ws/moderation", "MODERATION_TOKEN": "test-secret"}
        with patch.dict(os.environ, env, clear=True):
            settings = worker.Settings.from_env()
            self.assertEqual(settings.timeout, 30)
            self.assertEqual(settings.capacity, 16)
            self.assertEqual(settings.protocol, "bearer.test-secret")
            self.assertNotIn("test-secret", repr(settings))
        with patch.dict(os.environ, {**env, "CLEF_QUEUE_CAPACITY": "2", "CLEF_INFERENCE_TIMEOUT_SECONDS": "0.1"}, clear=True):
            settings = worker.Settings.from_env()
            self.assertEqual((settings.timeout, settings.capacity), (0.1, 2))
        for key, values in {
            "MODERATION_URL": ("ws://example.org/ws/moderation", "wss://user:password@example.org/ws/moderation",
                               "wss://example.org/ws/moderation?token=secret", "wss://example.org/wrong"),
            "MODERATION_TOKEN": ("", "bad token", "bad\nvalue"),
            "CLEF_INFERENCE_TIMEOUT_SECONDS": ("0", "-1", "nan", "inf", "bad"),
            "CLEF_QUEUE_CAPACITY": ("0", "-1", "1.5", "bad"),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), patch.dict(os.environ, {**env, key: value}, clear=True):
                    with self.assertRaises(ValueError):
                        worker.Settings.from_env()

    def test_result_contains_only_contract_fields(self):
        result = worker.result_message("id", {"decision": "allow", "categories": [],
                                               "reason": "ignored", "confidence": 1, "evidence": []})
        self.assertEqual(set(result), {"type", "comment_id", "worker_id", "decision", "categories"})
        self.assertEqual(result["worker_id"], "clef")
        for invalid in (None, {"decision": []}, {"decision": "error", "categories": []},
                        {"decision": "allow", "categories": ["url"]},
                        {"decision": "block", "categories": ["other"]},
                        {"decision": "block", "categories": [["url"]]},
                        {"decision": "block", "categories": ["url", "url"]}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                worker.result_message("id", invalid)


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def start(self, judge=None, capacity=16, timeout=30):
        self.ws = Socket()
        self.judge = judge or AsyncMock(return_value={"decision": "allow", "categories": []})
        self.task = asyncio.create_task(worker.session(
            self.ws, LivePolicy.load(), self.judge, worker.Settings("unused", "unused", timeout, capacity)))
        self.addAsyncCleanup(self.finish)

    async def finish(self):
        await self.ws.incoming.put(None)
        if self.task.done():
            if not self.task.cancelled(): self.task.exception()
        else:
            await asyncio.wait_for(self.task, 2)

    async def test_three_decisions_and_static_rule(self):
        judge = AsyncMock(side_effect=[{"decision": d, "categories": []} for d in ("allow", "block", "review")])
        await self.start(judge)
        for index, decision in enumerate(("allow", "block", "review")):
            await self.ws.push({**TASK, "comment_id": str(index)})
            result = await self.ws.result()
            self.assertEqual((result["comment_id"], result["decision"], result["categories"]), (str(index), decision, []))
        await self.ws.push({**TASK, "comment_id": "static", "text": "悠仁さま"})
        self.assertEqual((await self.ws.result())["categories"], ["imperial_topic"])
        self.assertEqual(judge.await_count, 3)

    async def test_invalid_inputs_review_and_unidentifiable_messages_ignored(self):
        await self.start()
        await self.ws.incoming.put("not json")
        await self.ws.incoming.put('{"type":"moderation.task","comment_id":"nan","text":NaN}')
        for task in ([], {**TASK, "comment_id": None}, {**TASK, "comment_id": ""},
                     {"type": "moderation.result", "comment_id": "notification"},
                     {"type": "error", "code": "already_decided", "comment_id": "conflict"}):
            await self.ws.push(task)
        for override in ({"text": None}, {"text": ""}, {"text": "あ" * 21}, {"text": "\ud800"},
                         {"text": "line\nfeed"}, {"channel_id": []}, {"channel_id": "unknown"}):
            await self.ws.push({**TASK, **override})
            self.assertEqual(await self.ws.result(), worker.review_message("first"))
        self.judge.assert_not_awaited()
        self.assertTrue(self.ws.outgoing.empty())

    async def test_recoverable_error_and_bad_output_then_next_task(self):
        judge = AsyncMock(side_effect=[ValueError("raw-comment-must-not-leak"),
                                      {"decision": "invalid", "categories": []},
                                      {"decision": "allow", "categories": ["url"]},
                                      {"decision": "allow", "categories": []}])
        await self.start(judge)
        with self.assertLogs("clef2", level="INFO") as logs:
            for index, decision in enumerate(("review", "review", "review", "allow")):
                await self.ws.push({**TASK, "comment_id": str(index)})
                self.assertEqual((await self.ws.result())["decision"], decision)
        logged = "\n".join(logs.output)
        self.assertNotIn("raw-comment-must-not-leak", logged)
        self.assertNotIn("がんばれ", logged)

    async def test_full_queue_reviews_new_task_and_keeps_fifo(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def judge(text):
            started.set()
            await release.wait()
            return {"decision": "allow", "categories": []}
        await self.start(judge, capacity=1)
        await self.ws.push(TASK)
        await started.wait()
        await self.ws.push({**TASK, "comment_id": "second"})
        await self.ws.push({**TASK, "comment_id": "overflow"})
        self.assertEqual(await self.ws.result(), worker.review_message("overflow"))
        release.set()
        for comment_id in ("first", "second"):
            result = await self.ws.result()
            self.assertEqual((result["comment_id"], result["decision"]), (comment_id, "allow"))

    async def test_timeout_reviews_current_and_waiting_then_exits(self):
        started = asyncio.Event()
        async def judge(text):
            started.set()
            await asyncio.Future()
        await self.start(judge, timeout=0.05)
        await self.ws.push(TASK)
        await started.wait()
        await self.ws.push({**TASK, "comment_id": "second"})
        self.assertEqual(await self.ws.result(), worker.review_message("first"))
        self.assertEqual(await self.ws.result(), worker.review_message("second"))
        with self.assertRaises(FatalInference):
            await self.task

    async def test_send_failure_does_not_resend_or_replace_attempted_decision(self):
        await self.start()
        self.ws.fail_send = True
        await self.ws.push(TASK)
        with self.assertRaises(OSError):
            await self.task
        self.assertEqual(self.ws.attempts, [worker.result_message("first", {"decision": "allow", "categories": []})])

    async def test_shutdown_review_has_bounded_send_time(self):
        started = asyncio.Event()
        async def judge(text):
            started.set()
            await asyncio.Future()
        await self.start(judge, timeout=0.02)
        async def blocked_send(raw): await asyncio.Future()
        self.ws.send = blocked_send
        with patch.object(worker, "SEND_TIMEOUT", 0.03):
            await self.ws.push(TASK)
            await started.wait()
            with self.assertRaises(FatalInference):
                await asyncio.wait_for(self.task, 0.5)

    async def test_log_escapes_comment_id(self):
        await self.start()
        with self.assertLogs("clef2", level="INFO") as logs:
            await self.ws.push({**TASK, "comment_id": "id\nforged-log"})
            await self.ws.result()
        self.assertTrue(all("\n" not in line for line in logs.output))


class WebSocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_reconnection_reuses_model_and_preserves_settings(self):
        sockets = [Socket(), Socket()]
        for index, ws in enumerate(sockets):
            await ws.push({**TASK, "comment_id": str(index)})
            # Keep the socket open until its reply has been sent.
        async def connections(*args, **kwargs):
            self.assertEqual(kwargs["subprotocols"], ["bearer.test-secret"])
            self.assertEqual(kwargs["max_queue"], 16)
            for ws in sockets:
                async def finish_after_reply(socket=ws):
                    await socket.outgoing.get()
                    await socket.incoming.put(None)
                closer = asyncio.create_task(finish_after_reply())
                yield ws
                await closer
        instance = SimpleNamespace(decide=AsyncMock(return_value={"decision": "allow", "categories": []}), close=Mock())
        with patch.object(worker, "Model", return_value=instance) as factory, \
                patch.object(worker, "connect", side_effect=connections), \
                patch.object(worker.asyncio, "sleep", new=AsyncMock()):
            await worker.run(worker.Settings("wss://example.org/ws/moderation", "bearer.test-secret"))
        factory.assert_called_once()
        instance.close.assert_called_once()
        self.assertEqual(instance.decide.await_count, 2)

    async def test_authentication_notifications_error_review_and_result(self):
        received = asyncio.get_running_loop().create_future()
        protocol = "bearer.test-secret"
        async def authenticate(connection, request):
            if request.headers.get("Sec-WebSocket-Protocol") != protocol:
                return connection.respond(403, "forbidden\n")
        async def server(ws):
            try:
                self.assertEqual(ws.subprotocol, protocol)
                await ws.send(json.dumps({"type": "moderation.result", "comment_id": "held"}))
                await ws.send(json.dumps(TASK))
                first = json.loads(await ws.recv())
                await ws.send(json.dumps({"type": "error", "code": "already_decided", "comment_id": "first"}))
                await ws.send(json.dumps({**TASK, "comment_id": "second"}))
                second = json.loads(await ws.recv())
                received.set_result((first, second))
            except Exception as exc:
                received.set_exception(exc)
            finally:
                await ws.close()
        judge = AsyncMock(side_effect=[ValueError("failed"), {"decision": "allow", "categories": []}])
        async with serve(server, "127.0.0.1", 0, subprotocols=[protocol], process_request=authenticate) as listener:
            url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}/ws/moderation"
            with self.assertRaises(InvalidStatus):
                async with connect(url, subprotocols=["bearer.wrong"], proxy=None):
                    pass
            async with connect(url, subprotocols=[protocol], proxy=None) as ws:
                await asyncio.wait_for(worker.session(ws, LivePolicy.load(), judge, worker.Settings(url, protocol)), 3)
            first, second = await received
        self.assertEqual(first, worker.review_message("first"))
        self.assertEqual(second["decision"], "allow")
        self.assertEqual(judge.await_count, 2)


if __name__ == "__main__":
    unittest.main()
