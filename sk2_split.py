"""
sk2_split.py  --  article-grouped train/test split (Stage 1 fix, 26 Oct 2026)

WHY THIS FILE EXISTS
--------------------
The old split (sk2_retrieval.make_split) shuffled RULE ids. Many SigmaHQ rules
cite the same article, so sibling rules of one report landed on both sides of
the split: 84 of the 216 evaluated test instances in results_core2 shared their
article with the exemplar pool (78 by URL, 84 counting identical text). The
TF-IDF retriever then found the identical training article and showed its
sibling expert rule as an exemplar, and the nn baseline copied it.

This module puts every rule of one article into the SAME split.

WHAT COUNTS AS "THE SAME ARTICLE"  (union of three tests)
  1. same normalised URL   (scheme, www., trailing slash, #fragment, and
                            tracking query parameters removed; case-folded)
  2. identical normalised text  (whitespace collapsed, first 20,000 chars,
                                 the same window the retriever reads)
  3. near-duplicate text   word 5-shingle Jaccard >= NEAR_DUP_JACCARD
                           (mirrors, reposts, AMP pages)

Pure Python, no sklearn, so it can be checked on any machine.
"""

from __future__ import annotations

import hashlib
import zlib
import json
import random
import re
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode

GROUPED_SPLIT_SEED = 20261026     # NEW seed: the old split (20260823) was
                                  # evaluated several times and is exploratory
NEAR_DUP_JACCARD = 0.80
TEXT_WINDOW = 20000
_TRACKING = re.compile(r"^(utm_|fbclid|gclid|mc_|ref$|source$)", re.I)


def norm_url(u: str) -> str:
    u = str(u or "").strip()
    if not u:
        return ""
    try:
        p = urlsplit(u)
    except ValueError:
        return u.lower()
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+$", "", p.path or "")
    path = re.sub(r"/amp$", "", path)
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
         if not _TRACKING.match(k)]
    q = ("?" + urlencode(sorted(q))) if q else ""
    return f"{host}{path}{q}".lower()


def norm_text(t: str) -> str:
    t = re.sub(r"<[^>]+>", " ", str(t or ""))
    return re.sub(r"\s+", " ", t).strip().lower()[:TEXT_WINDOW]


def _shingles(t: str, n: int = 5) -> set:
    w = t.split()
    # zlib.crc32, not hash(): Python randomises str hashes per process, which
    # would make the near-duplicate groups (and so the split) non-deterministic
    return {zlib.crc32(" ".join(w[i:i + n]).encode("utf-8", "ignore"))
            for i in range(max(0, len(w) - n + 1))}


class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def article_groups(ids, url_of: dict, text_of: dict, near_dup: bool = True):
    """
    -> (group_of, report)
       group_of: id -> group key (the smallest id in its group)
       report:   counts of merges by each test, for the run log
    """
    ids = sorted(ids)
    uf = _UF(ids)
    rep = {"by_url": 0, "by_text": 0, "by_near_dup": 0}

    first = {}
    for i in ids:
        k = norm_url(url_of.get(i, ""))
        if k:
            if k in first and uf.find(first[k]) != uf.find(i):
                uf.union(first[k], i); rep["by_url"] += 1
            first.setdefault(k, i)

    texts = {i: norm_text(text_of.get(i, "")) for i in ids}
    first = {}
    for i in ids:
        if not texts[i]:
            continue
        h = hashlib.md5(texts[i].encode("utf-8", "ignore")).hexdigest()
        if h in first and uf.find(first[h]) != uf.find(i):
            uf.union(first[h], i); rep["by_text"] += 1
        first.setdefault(h, i)

    if near_dup:
        # candidate pairs: share >= 8 of their 64 smallest shingle hashes
        sh = {i: _shingles(texts[i]) for i in ids if texts[i]}
        sig = {i: set(sorted(s)[:64]) for i, s in sh.items() if len(s) >= 20}
        inv = {}
        for i, s in sig.items():
            for x in s:
                inv.setdefault(x, []).append(i)
        cand = {}
        for lst in inv.values():
            if len(lst) > 50:
                continue
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    key = (lst[a], lst[b])
                    cand[key] = cand.get(key, 0) + 1
        for (a, b), c in cand.items():
            if c < 8 or uf.find(a) == uf.find(b):
                continue
            j = len(sh[a] & sh[b]) / max(1, len(sh[a] | sh[b]))
            if j >= NEAR_DUP_JACCARD:
                uf.union(a, b); rep["by_near_dup"] += 1

    group_of = {i: uf.find(i) for i in ids}
    rep["n_ids"] = len(ids)
    rep["n_groups"] = len(set(group_of.values()))
    rep["n_multi_rule_groups"] = sum(
        1 for g in set(group_of.values())
        if sum(1 for v in group_of.values() if v == g) > 1)
    return group_of, rep


def grouped_split(ids, group_of: dict, test_frac: float = 0.30,
                  seed: int = GROUPED_SPLIT_SEED):
    """Shuffle GROUPS, fill test until it holds ~test_frac of the ids."""
    ids = sorted(ids)
    groups = {}
    for i in ids:
        groups.setdefault(group_of[i], []).append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = int(round(len(ids) * test_frac))
    test, train = [], []
    for g in keys:
        (test if len(test) < target else train).extend(groups[g])
    split = {"seed": seed, "test_frac": test_frac, "grouped_by": "article",
             "near_dup_jaccard": NEAR_DUP_JACCARD,
             "train": sorted(train), "test": sorted(test)}
    assert_no_group_overlap(split["train"], split["test"], group_of)
    return split


def carve_dev(train_ids, group_of: dict, dev_frac: float, seed: int):
    """Dev set carved from TRAIN by group, so dev and exemplar pool never share an article."""
    sub = grouped_split(train_ids, group_of, test_frac=dev_frac, seed=seed)
    return sub["test"], sub["train"]


def assert_no_group_overlap(a_ids, b_ids, group_of: dict):
    ga = {group_of[i] for i in a_ids if i in group_of}
    shared = sorted(i for i in b_ids if group_of.get(i) in ga)
    if shared:
        raise ValueError(f"ARTICLE LEAKAGE: {len(shared)} ids share an article "
                         f"across the split, e.g. {shared[:3]}")


def groups_from_bench(bench, ids):
    """Read URLs from index.json and text from articles/{id}.txt."""
    bench = Path(bench)
    index = json.loads((bench / "index.json").read_text(encoding="utf-8"))
    url_of = {r["id"]: r.get("url", "") for r in index}
    text_of = {}
    for i in ids:
        try:
            text_of[i] = (bench / "articles" / f"{i}.txt").read_text(
                encoding="utf-8", errors="ignore")
        except OSError:
            text_of[i] = ""
    return article_groups(ids, url_of, text_of)
