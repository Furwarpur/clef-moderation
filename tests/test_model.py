import asyncio
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import model
from policy import LivePolicy, ROOT


class ModelTests(unittest.TestCase):
    def test_payload_keeps_original_text_and_single_decision(self):
        text = "応援\u200dしています"
        policy = LivePolicy.load()
        payload = model.decision_payload(text, policy.text)
        self.assertEqual(payload["state"], {"comment": text})
        self.assertEqual(list(payload["questions"]), ["decision"])
        self.assertEqual(list(payload["questions"]["decision"]["criteria"]), ["review", "block", "allow"])
        self.assertTrue(payload["questions"]["decision"]["instructions"].endswith(policy.text))

    def test_validate_model_probabilities(self):
        for decision in ("allow", "block", "review"):
            answer = {"type": "choice", "choice": decision,
                      "probabilities": {k: int(k == decision) for k in model.LABELS}}
            self.assertEqual(model.validate_answer(answer), {"decision": decision, "categories": []})
        cases = [None, {}, {"type": "choice", "choice": "error", "probabilities": {}},
                 {"type": "choice", "choice": [], "probabilities": {}},
                 {"type": "choice", "choice": "allow", "probabilities": {"allow": 0, "block": 1, "review": 0}}]
        for value in (math.nan, math.inf, -1, 2, True, "1"):
            cases.append({"type": "choice", "choice": "allow",
                          "probabilities": {"allow": value, "block": 0, "review": 0}})
        for answer in cases:
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                model.validate_answer(answer)

    def test_full_input_encoded_before_2048_limit_check(self):
        instance = model.Model.__new__(model.Model)
        instance.policy = LivePolicy.load()
        instance.tokenizer = object()
        instance.release = SimpleNamespace(encode_record=Mock())
        instance.release.encode_record.return_value = SimpleNamespace(input_ids=[1] * 2048)
        instance.encode("がんばれ")
        self.assertGreater(instance.release.encode_record.call_args.kwargs["max_length"], 2048)
        instance.release.encode_record.return_value = SimpleNamespace(input_ids=[1] * 2049)
        with self.assertRaisesRegex(ValueError, "model_input_limit"):
            instance.encode("がんばれ")

    def test_configuration_does_not_force_tuning_off(self):
        for enabled in ("0", "1"):
            with patch.dict(os.environ, {"PYTORCH_TUNABLEOP_TUNING": enabled,
                                         "PYTORCH_TUNABLEOP_FILENAME": str(ROOT / "runtime/tunableop.csv")}, clear=True):
                model.configure_environment()
                self.assertEqual(os.environ["PYTORCH_TUNABLEOP_TUNING"], enabled)
                self.assertEqual(os.environ["PYTORCH_TUNABLEOP_ENABLED"], "1")
                self.assertEqual(os.environ["PYTORCH_TUNABLEOP_RECORD_UNTUNED"], "0")
        for env in ({"PYTORCH_TUNABLEOP_TUNING": "yes"},
                    {"PYTORCH_TUNABLEOP_FILENAME": str(ROOT / "runtime/tunableop-original.csv")},
                    {"PYTORCH_TUNABLEOP_FILENAME": "/tmp/tuned.csv"}):
            with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                model.configure_environment()

    def test_runtime_integrity_and_missing_csv(self):
        with patch.dict(os.environ, {}, clear=True):
            model.check_runtime()
        with patch.dict(os.environ, {"PYTORCH_TUNABLEOP_FILENAME": str(ROOT / "runtime/absent.csv")}, clear=True):
            with self.assertRaises(FileNotFoundError):
                model.check_runtime()
        manifest = json.loads((ROOT / "manifest.json").read_text())
        manifest["packages"]["torch"] = "wrong"
        with patch("model.json.loads", return_value=manifest), self.assertRaisesRegex(ValueError, "package_version"):
            model.check_runtime()

    def test_relative_tuning_path_uses_project_root_regardless_of_cwd(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                for value in ("runtime/tunableop.csv", str(ROOT / "runtime/tunableop.csv")):
                    with patch.dict(os.environ, {"PYTORCH_TUNABLEOP_FILENAME": value}, clear=True):
                        self.assertEqual(model.configure_environment(), ROOT / "runtime/tunableop.csv")
                        self.assertEqual(os.environ["PYTORCH_TUNABLEOP_FILENAME"], str(ROOT / "runtime/tunableop.csv"))
                for value in ("../tuned.csv", "runtime/../../tuned.csv", "runtime/tunableop-original.csv", ""):
                    with self.subTest(value=value), \
                            patch.dict(os.environ, {"PYTORCH_TUNABLEOP_FILENAME": value}, clear=True), \
                            self.assertRaises(ValueError):
                        model.configure_environment()
            finally:
                os.chdir(previous)

    def test_atomic_save_preserves_existing_file_on_failure(self):
        tunable = SimpleNamespace(get_validators=lambda: (("PT_VERSION", "2.12.0"),),
                                  get_results=lambda: (("Gemm", "shape", "solution", 1.0),))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tuned.csv"
            path.write_text("existing")
            with patch("model.Path.replace", side_effect=OSError("disk failure")), self.assertRaises(OSError):
                model.save_tuning(tunable, path)
            self.assertEqual(path.read_text(), "existing")
            self.assertEqual(list(path.parent.iterdir()), [path])
            model.save_tuning(tunable, path)
            with path.open(newline="") as stream:
                self.assertEqual(list(csv.reader(stream)),
                                 [["Validator", "PT_VERSION", "2.12.0"], ["Gemm", "shape", "solution", "1.0"]])

    def test_tuning_hook_saves_new_results_and_reuses_when_disabled(self):
        class Tunable:
            def __init__(self):
                self.rows = []
            def enable(self, value): self.enabled = value
            def tuning_enable(self, value): self.tuning = value
            def record_untuned_enable(self, value): self.recording = value
            def set_filename(self, path): self.filename = path
            def read_file(self, path):
                with open(path, newline="") as stream:
                    self.rows = [tuple(row) for row in csv.reader(stream) if row and row[0] != "Validator"]
                return True
            def get_results(self): return tuple(self.rows)
            def get_validators(self): return (("PT_VERSION", "2.12.0"),)
            def is_enabled(self): return self.enabled
            def tuning_is_enabled(self): return self.tuning

        tunable = Tunable()
        torch = SimpleNamespace(cuda=SimpleNamespace(tunable=tunable))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tuned.csv"
            path.write_text("Validator,PT_VERSION,2.12.0\nGemm,old,solution,1\n")
            for switch in ("1", "0"):
                class Head:
                    def load_weights(self): return "loaded"
                    def score(self):
                        if tunable.tuning and not any(row[1] == "new" for row in tunable.rows):
                            tunable.rows.append(("Gemm", "new", "faster", "0.5"))
                        return "scored"
                env = {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": switch,
                       "PYTORCH_TUNABLEOP_FILENAME": str(path)}
                with patch.dict("sys.modules", {"torch": torch}), patch.dict(os.environ, env):
                    model.install_tunable(Head, object)
                    head = Head()
                    self.assertEqual(head.load_weights(), "loaded")
                    self.assertEqual(head.score(), "scored")
                    self.assertEqual(len(tunable.rows), 2)
                    self.assertNotEqual(tunable.filename, str(path))
                    self.assertFalse(tunable.recording)
            self.assertIn("Gemm,new,faster,0.5", path.read_text())


class InferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_exactly_one_engine_call_and_review_is_preserved(self):
        class Flat:
            ndim = 1
            def __len__(self): return 3
            def float(self): return self
            def softmax(self, dimension): return self
            def tolist(self): return [1.0, 0.0, 0.0]

        calls = []
        async def encode(*args):
            calls.append(args)
            yield SimpleNamespace(outputs=SimpleNamespace(data=Flat()))

        instance = model.Model.__new__(model.Model)
        instance.policy = LivePolicy.load()
        instance.tokenizer = object()
        encoded = SimpleNamespace(input_ids=[1, 2], questions=[SimpleNamespace(option_ids=("review", "block", "allow"))])
        instance.release = SimpleNamespace(encode_record=Mock(return_value=encoded),
            systemone_answer=lambda question, probs: {"type": "choice", "choice": "review", "probabilities": probs})
        instance.engine = SimpleNamespace(encode=encode, errored=False)
        modules = {"vllm": SimpleNamespace(PoolingParams=lambda **kw: kw),
                   "clef_vllm.schema": SimpleNamespace(TASK="token_classify", schema_layout=lambda encoded: [])}
        with patch.dict("sys.modules", modules):
            self.assertEqual(await instance.decide("がんばれ"), {"decision": "review", "categories": []})
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
