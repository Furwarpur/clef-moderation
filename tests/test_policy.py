import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from policy import LivePolicy, ROOT, normalize_for_matching, validate_text


class PolicyTests(unittest.TestCase):
    def test_all_49_cases_match_existing_static_results(self):
        path = ROOT / "cases-live-v1.jsonl"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "4f796321705113c094413a7e40bbd2d5b27ebae7ec51fa63acd296b6d5350930")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        golden = json.loads((ROOT / "tests/static-results.json").read_text())
        policy = LivePolicy.load()
        self.assertEqual(len(rows), 49)
        self.assertEqual(sum(row["expected_static"] == "block" for row in rows), 25)
        for row in rows:
            with self.subTest(case=row["id"]):
                self.assertEqual(policy.static_result(row["text"]), golden[row["id"]])

    def test_matching_preserves_normalization_and_category_order(self):
        policy = LivePolicy.load()
        for text in ("悠\u200b仁", "ﾋｻﾋﾄ", "ＨＩＳＡＨＩＴＯ", "ひ・さ・ひ・と"):
            self.assertEqual(policy.static_result(text)["categories"], ["imperial_topic"])
        self.assertEqual(policy.static_result("悠仁 核実験 www.a.jp")["categories"],
                         ["imperial_topic", "nuclear_topic", "url"])
        for text in ("核酸について勉強中", "皇帝ペンギンかわいい", "永田さん", "応援\u200dしています"):
            self.assertIsNone(policy.static_result(text))

    def test_input_boundaries_and_unicode(self):
        for text in ("あ" * 20, "😀" * 20, "がんばれ", "応援\u200dしています"):
            validate_text(text)
        for text in (None, [], 1, "", " \u3000", "あ" * 21, "が\nんばれ", "\0", "\ud800", "\x7f"):
            with self.subTest(text=ascii(text)), self.assertRaises(ValueError):
                validate_text(text)

    def test_changed_policy_or_rules_prevent_startup(self):
        for changed in ("policy-live-v1.md", "live-policy-v1.json"):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                for name in ("policy-live-v1.md", "live-policy-v1.json"):
                    (root / name).write_bytes((ROOT / name).read_bytes())
                with (root / changed).open("ab") as stream:
                    stream.write(b" ")
                with self.assertRaises(ValueError):
                    LivePolicy.load(root)


if __name__ == "__main__":
    unittest.main()
