import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from compare_gpu import cases, measure, record_baseline, socket_smoke, statistics_ms
from policy import ROOT


class ComparisonTests(unittest.IsolatedAsyncioTestCase):
    async def test_measure_separates_first_warmup_and_three_repetitions(self):
        calls = []
        async def judge(text):
            calls.append(text)
            return {"decision": "allow", "categories": []}
        rows = [{"id": "a", "text": "a"}, {"id": "b", "text": "b"}]
        result = await measure(judge, rows)
        self.assertEqual(len(calls), 1 + 2 + 3 * 2)
        self.assertEqual(result["first_request"]["id"], "a")
        self.assertEqual([(r["id"], r["repeat"]) for r in result["rows"]],
                         [("a", 0), ("b", 0), ("a", 1), ("b", 1), ("a", 2), ("b", 2)])
        self.assertEqual(statistics_ms([{"elapsed_ms": n} for n in (1, 2, 3, 4, 5)]),
                         {"median_ms": 3, "max_ms": 5, "p95_ms": 4.8})

    async def test_socket_smoke_checks_all_cases_and_counts_engine_calls(self):
        class Model:
            engine = SimpleNamespace(encode=Mock())
            async def decide(self, text):
                self.engine.encode(text)
                return {"decision": "allow", "categories": []}
        baseline = {"rows": [{"id": row["id"], "decision": "allow"} for row in cases()]}
        self.assertEqual(await socket_smoke(Model(), baseline),
                         {"cases": 49, "engine_calls": 24, "invalid_input_review": True})


class BaselineTests(unittest.TestCase):
    def test_verified_old_entrypoint_runs_without_starting_http(self):
        async def startup(): pass
        async def shutdown(): pass
        requests = []
        async def endpoint(payload):
            requests.append(payload)
            return {"answers": {"decision": {"type": "choice", "choice": "allow",
                                               "probabilities": {"allow": 1, "block": 0, "review": 0}}}}
        app = SimpleNamespace(router=SimpleNamespace(on_startup=[startup], on_shutdown=[shutdown]),
                              routes=[SimpleNamespace(path="/v1/systemone", endpoint=endpoint)])
        fake_uvicorn = SimpleNamespace(run=Mock())
        def entrypoint(path, run_name):
            fake_uvicorn.run(app, host="127.0.0.1", port=11634)
        config = {"command": ["python", "verified-old-entrypoint.py"], "environment": {
            "PYTORCH_TUNABLEOP_FILENAME": str(ROOT / "runtime/tunableop.csv"), "PYTHONPATH": str(ROOT)}}
        modules = {"uvicorn": fake_uvicorn, "serve_clef_vllm": SimpleNamespace(prepared_configuration=lambda: config)}
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"PYTORCH_TUNABLEOP_FILENAME": config["environment"]["PYTORCH_TUNABLEOP_FILENAME"]}, clear=True), \
                patch.object(sys, "path", list(sys.path)), patch.object(sys, "argv", []), \
                patch.dict(sys.modules, modules), patch("compare_gpu.check_runtime"), \
                patch("compare_gpu.check_api_stopped") as stopped, patch("compare_gpu.runpy.run_path", side_effect=entrypoint):
            output = Path(folder) / "baseline.json"
            self.assertEqual(record_baseline(SimpleNamespace(output=output)), 0)
            stopped.assert_called_once()
            result = json.loads(output.read_text())
            self.assertEqual(len(result["rows"]), 72)
            self.assertEqual(len(requests), 1 + 24 + 72)
            self.assertTrue(all(list(p["questions"]) == ["decision"] for p in requests))
