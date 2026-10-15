"""Assert real GPU tuning is persisted and reused by a fresh tuning-disabled engine."""
import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import Model, digest, validate_tuning_file
from policy import LivePolicy, ROOT
from compare_gpu import check_api_stopped


def result_count(path):
    with path.open(newline="") as stream:
        return sum(1 for row in csv.reader(stream) if row and row[0] != "Validator")


async def main(mode, resume=False):
    check_api_stopped()
    path = ROOT / "runtime/tunableop-validation.csv"
    report_path = ROOT / "runtime/tuning-validation.json"
    original_hash = digest(ROOT / "runtime/tunableop-original.csv")
    if mode == "on":
        if not resume:
            shutil.copyfile(ROOT / "runtime/tunableop.csv", path)
        before = result_count(path)
        # Bound this validation run; production tuning keeps PyTorch's normal limits.
        os.environ["PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS"] = "1" if resume else "5"
        os.environ["PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS"] = "1" if resume else "4"
    else:
        report = json.loads(report_path.read_text())
        assert digest(path) == report["saved_sha256"]
        before = result_count(path)
    os.environ.update(PYTORCH_TUNABLEOP_ENABLED="1", PYTORCH_TUNABLEOP_TUNING="1" if mode == "on" else "0",
                      PYTORCH_TUNABLEOP_FILENAME=str(path))
    started = time.perf_counter()
    instance = Model(LivePolicy.load())
    startup_s = time.perf_counter() - started
    try:
        started = time.perf_counter()
        async with asyncio.timeout(300):
            decision = await instance.decide("がんばれ")
        elapsed_s = time.perf_counter() - started
        validate_tuning_file(path)
        after = result_count(path)
        if mode == "on":
            assert after > before, (before, after)
            report = {"before": before, "after": after, "saved_sha256": digest(path),
                      "decision": decision["decision"], "tuning_startup_s": startup_s,
                      "tuning_inference_s": elapsed_s, "test_max_ms_per_solution": 1 if resume else 5,
                      "test_max_iterations": 1 if resume else 4}
        else:
            assert after == before and digest(path) == report["saved_sha256"]
            assert decision["decision"] == report["decision"]
            report.update(reused_after_restart=True, tuning_off_inference_s=elapsed_s, tuning_off_startup_s=startup_s)
        assert digest(ROOT / "runtime/tunableop-original.csv") == original_hash
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        instance.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("on", "off"))
    parser.add_argument("--resume", action="store_true", help="reuse results saved by an interrupted tuning test")
    args = parser.parse_args()
    asyncio.run(main(args.mode, args.resume))
