"""Pinned live policy and the existing deterministic rejection rules."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
POLICY_SHA256 = "aaa79d937e598d3ad4546355d6f7fe9b700885f0f3a807d60eab0904d0be0d4c"
RULES_SHA256 = "b7be87b08edfaf3f3ad05128bd12ae3ece8b52d44de8999d87cf8e3dfe67adfa"
CATEGORIES = ("tsukuba_person", "imperial_topic", "nuclear_topic", "url")
URL_PATTERN = re.compile(
    r"(?:https?|ftp|wss?|file)\s*:\s*[/\\]{2}|www\s*\.|"
    r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:xn--[a-z0-9-]+|[a-z]{2,63})(?![a-z0-9-])",
    re.IGNORECASE,
)


def normalize_for_matching(text, *, compact=False):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)
    return "".join(c for c in text if unicodedata.category(c) != "Cf"
                   and (not compact or (not c.isspace() and not unicodedata.category(c).startswith("P"))))


def validate_text(text):
    if (not isinstance(text, str) or not text.strip() or not 1 <= len(text) <= 20
            or any(unicodedata.category(c) in {"Cc", "Cs"} for c in text)):
        raise ValueError("invalid_text")


@dataclass(frozen=True)
class LivePolicy:
    text: str
    groups: tuple

    @classmethod
    def load(cls, root=ROOT):
        raw = (root / "live-policy-v1.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != RULES_SHA256:
            raise ValueError("static_rules_hash_mismatch")
        config = json.loads(raw)
        if (config.get("schema_version") != 1 or config.get("policy_id") != "tsukuba-live-v1"
                or config.get("policy_file") != "policy-live-v1.md"
                or config.get("policy_sha256") != POLICY_SHA256):
            raise ValueError("invalid_policy_config")
        policy = (root / config["policy_file"]).read_bytes()
        if hashlib.sha256(policy).hexdigest() != POLICY_SHA256:
            raise ValueError("policy_hash_mismatch")
        text = policy.decode("utf-8")
        if "policy_id: tsukuba-live-v1" not in text:
            raise ValueError("policy_id_mismatch")
        groups = []
        for group in config["static_groups"]:
            terms = tuple(normalize_for_matching(t, compact=True) for t in group["terms"])
            if not terms or any(not term for term in terms):
                raise ValueError("invalid_static_terms")
            groups.append((group["category"], terms))
        if tuple(category for category, _ in groups) != CATEGORIES[:-1]:
            raise ValueError("invalid_static_categories")
        return cls(text, tuple(groups))

    def static_result(self, text):
        validate_text(text)
        match_text = normalize_for_matching(text, compact=True)
        categories = [category for category, terms in self.groups if any(t in match_text for t in terms)]
        url_text = normalize_for_matching(text)
        if (URL_PATTERN.search(url_text)
                or URL_PATTERN.search("".join(c for c in url_text if not c.isspace()))):
            categories.append("url")
        return {"decision": "block", "categories": categories} if categories else None
