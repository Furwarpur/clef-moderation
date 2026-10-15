"""Compare the old implementation and clef2 sequentially without opening an HTTP API."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import runpy
import statistics
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import Model, check_runtime, decision_payload, digest, validate_answer
from policy import LivePolicy, ROOT


def statistics_ms(rows):
    values = [row["elapsed_ms"] for row in rows]
    return {"median_ms": statistics.median(values), "max_ms": max(values),
            "p95_ms": statistics.quantiles(values, n=20, method="inclusive")[18]}


async def measure(judge, rows):
    async def once(row):
        began = time.perf_counter()
        result = await judge(row["text"])
        return {"id": row["id"], "decision": result["decision"],
                "elapsed_ms": (time.perf_counter() - began) * 1000}
    first = await once(rows[0])
    for row in rows:
        await once(row)
    results = []
    for repeat in range(3):
        for row in rows:
            results.append({**await once(row), "repeat": repeat})
    return {"first_request": first, "rows": results, "statistics": statistics_ms(results)}


async def socket_smoke(instance, baseline):
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve
    from worker import Settings, session
    policy = LivePolicy.load()
    rows = [json.loads(line) for line in (ROOT / "cases-live-v1.jsonl").read_text().splitlines()]
    expected = {row["id"]: row["decision"] for row in baseline["rows"]}
    calls = 0
    original_encode = instance.engine.encode
    def counted_encode(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_encode(*args, **kwargs)
    instance.engine.encode = counted_encode
    completed = asyncio.get_running_loop().create_future()
    async def server(ws):
        try:
            assert ws.subprotocol == "bearer.local-gpu-test"
            for row in rows:
                before = calls
                await ws.send(json.dumps({"type": "moderation.task", "comment_id": row["id"],
                                          "channel_id": "uni", "text": row["text"]}))
                reply = json.loads(await ws.recv())
                assert reply["comment_id"] == row["id"]
                assert set(reply) == {"type", "comment_id", "worker_id", "decision", "categories"}
                assert reply["decision"] == ("block" if row["expected_static"] else expected[row["id"]])
                assert calls - before == (0 if row["expected_static"] else 1)
            await ws.send(json.dumps({"type": "moderation.task", "comment_id": "invalid-input",
                                      "channel_id": "uni", "text": None}))
            reply = json.loads(await ws.recv())
            assert reply["decision"] == "review" and reply["categories"] == []
            completed.set_result({"cases": len(rows), "engine_calls": calls, "invalid_input_review": True})
        except Exception as exc:
            completed.set_exception(exc)
        finally:
            await ws.close()
    try:
        async with serve(server, "127.0.0.1", 0, subprotocols=["bearer.local-gpu-test"]) as listener:
            url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}/ws/moderation"
            async with connect(url, subprotocols=["bearer.local-gpu-test"], proxy=None) as ws:
                await session(ws, policy, instance.decide, Settings(url, "bearer.local-gpu-test"))
            return await completed
    finally:
        instance.engine.encode = original_encode


def check_api_stopped():
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 11634)) == 0:
            raise ValueError("stop the old GPU service before measuring")


def cases():
    rows = [json.loads(line) for line in (ROOT / "cases-live-v1.jsonl").read_text().splitlines()]
    return [row for row in rows if row["expected_static"] is None]


def save_report(args, result):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "statistics": result["statistics"],
                      "passed": result.get("passed")}, ensure_ascii=False), flush=True)
    return 0 if result.get("passed", True) else 1


def record_baseline(args):
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "0"
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    check_runtime()
    check_api_stopped()
    policy = LivePolicy.load()
    working_csv = Path(os.environ["PYTORCH_TUNABLEOP_FILENAME"])
    sys.path.insert(0, str(ROOT.parent / "moderation-harness"))
    from serve_clef_vllm import prepared_configuration
    config = prepared_configuration()
    expected = digest(Path(config["environment"]["PYTORCH_TUNABLEOP_FILENAME"]))
    if expected != digest(working_csv):
        raise ValueError("old implementation and clef2 must use the same CSV")
    os.environ.update(config["environment"])
    for path in reversed(config["environment"]["PYTHONPATH"].split(os.pathsep)):
        sys.path.insert(0, path)
    report = None

    def without_http(app, **kwargs):
        async def run():
            for callback in app.router.on_startup:
                await callback()
            endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", "") == "/v1/systemone")
            async def judge(text):
                payload = {"model": "clef-flash:9b-fp16-vllm-poc", **decision_payload(text, policy.text)}
                async with asyncio.timeout(30):
                    response = await endpoint(payload)
                return validate_answer(response["answers"]["decision"])
            try:
                return await measure(judge, cases())
            finally:
                for callback in app.router.on_shutdown:
                    await callback()
        nonlocal report
        report = asyncio.run(run())

    sys.argv = config["command"][1:]
    # Run the verified old entrypoint, replacing only its HTTP server launch.
    with patch("uvicorn.run", side_effect=without_http):
        runpy.run_path(sys.argv[0], run_name="__main__")
    assert report is not None
    report.update(tuning_sha256=expected, note="Old verified entrypoint, direct endpoint call; no HTTP listener.")
    return save_report(args, report)


async def main(args):
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "0"
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    check_runtime()
    policy = LivePolicy.load()
    baseline = json.loads(args.baseline.read_text())
    if baseline["tuning_sha256"] != digest(Path(os.environ["PYTORCH_TUNABLEOP_FILENAME"])):
        raise ValueError("CSV differs from the baseline")
    check_api_stopped()
    instance = Model(policy)
    try:
        async def judge(text):
            async with asyncio.timeout(30):
                return await instance.decide(text)
        result = await measure(judge, cases())
        result["socket_smoke"] = await socket_smoke(instance, baseline)
    finally:
        instance.close()
    if [(row["id"], row["repeat"]) for row in result["rows"]] != [
            (row["id"], row["repeat"]) for row in baseline["rows"]]:
        raise ValueError("baseline cases do not match")
    mismatches = [{"id": new["id"], "repeat": new["repeat"], "old": old["decision"], "new": new["decision"]}
                  for old, new in zip(baseline["rows"], result["rows"]) if old["decision"] != new["decision"]]
    ratio = result["statistics"]["median_ms"] / baseline["statistics"]["median_ms"]
    result.update(tuning_sha256=baseline["tuning_sha256"], mismatches=mismatches,
                  median_ratio=ratio, passed=not mismatches and ratio <= 1.1)
    return save_report(args, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "compare"))
    parser.add_argument("--baseline", type=Path, default=ROOT / "runtime/comparison-baseline.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "runtime" / ("comparison-baseline.json" if args.mode == "baseline" else "comparison.json")
    raise SystemExit(record_baseline(args) if args.mode == "baseline" else asyncio.run(main(args)))
