"""
sk2_retrieval.py  --  TASK 3

Retrieval-augmented exemplars.

The old prompt carried two fixed examples (Bitbucket and Mimikatz) for every
report on earth. A report about macOS network sniffing was shown an AWS audit
example. This module replaces them with the k most similar rules from the
TRAINING split, so the model sees the conventions that real authors use for
that kind of telemetry.

Leakage discipline, enforced in code, not by convention:
  * the index is built ONLY from ids passed as `train_ids`
  * `retrieve()` raises if the query id is present in the index
  * the retrieved neighbours are public SigmaHQ rules, never the target rule

TF-IDF over character n-grams + word unigrams. Deterministic, no API cost, no
model download, and reproducible from the committed split file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def _clean(t: str) -> str:
    t = re.sub(r"<[^>]+>", " ", str(t or ""))
    t = re.sub(r"\s+", " ", t)
    return t.strip()


class ExemplarIndex:
    """Nearest-neighbour retrieval over training CTI articles."""

    def __init__(self, docs: dict[str, str], rules: dict[str, dict], train_ids: list[str],
                 group_of: dict | None = None):
        self.train_ids = [i for i in train_ids if i in docs and i in rules]
        # Stage 1 fix (26 Oct 2026): article groups. When given, retrieve()
        # never returns a neighbour from the query's own article, even if a
        # caller builds the pool wrongly. Counted in self.same_article_skipped.
        self.group_of = group_of or {}
        self.same_article_skipped = 0
        if not self.train_ids:
            raise ValueError("ExemplarIndex: no usable training ids")
        self.rules = rules
        corpus = [_clean(docs[i])[:20000] for i in self.train_ids]

        # max_df=0.9 drops any term appearing in >90% of documents. On a small
        # or unusually uniform corpus that can prune EVERYTHING and raise
        # "After pruning, no terms remain". Fall back to no pruning rather than
        # taking the whole experiment down over a vectoriser setting.
        def _fit(**kw):
            v = TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                ngram_range=(1, 2), max_features=200_000, **kw)
            return v, v.fit_transform(corpus)

        try:
            self.vec, self.M = _fit(min_df=1, max_df=0.9, stop_words="english")
        except ValueError:
            try:
                self.vec, self.M = _fit(min_df=1, max_df=1.0, stop_words="english")
            except ValueError:
                self.vec, self.M = _fit(min_df=1, max_df=1.0)

    def retrieve(self, query_id: str | None, query_text: str, k: int = 5):
        if query_id is not None and query_id in self.train_ids:
            raise ValueError(
                f"LEAKAGE GUARD: '{query_id}' is in the training index. "
                "Never retrieve exemplars for a test item from a split containing it.")
        q = self.vec.transform([_clean(query_text)[:20000]])
        sims = linear_kernel(q, self.M).ravel()
        qg = self.group_of.get(query_id) if query_id is not None else None
        out = []
        for idx in np.argsort(-sims):
            rid = self.train_ids[int(idx)]
            if qg is not None and self.group_of.get(rid) == qg:
                self.same_article_skipped += 1
                continue
            out.append({"id": rid, "score": float(sims[int(idx)]), "rule": self.rules[rid]})
            if len(out) >= k:
                break
        return out


def exemplar_to_prompt(ex: dict) -> tuple[str, str]:
    """
    Render one retrieved neighbour as a (user, assistant) message pair in the
    same JSON shape the extractor is asked to produce.

    NOTE: the neighbour's own CTI text is NOT included -- only its rule. The
    exemplar teaches CONVENTION (which fields, which modifiers), not content.
    """
    r = ex["rule"]
    ls = r.get("logsource", {}) or {}
    det = (r.get("detection") or {})
    sel = {}
    for bname, block in det.items():
        if str(bname).lower() in ("condition", "timeframe"):
            continue
        if isinstance(block, dict):
            for k, v in block.items():
                sel[k] = v
        if len(sel) >= 6:
            break
    tags = [t for t in (r.get("tags") or []) if str(t).startswith("attack.")]

    user = ("EXAMPLE OF HOUSE STYLE FOR THIS TELEMETRY TYPE.\n"
            f"log source: product={ls.get('product')} "
            f"category={ls.get('category')} service={ls.get('service')}")
    assistant = json.dumps({
        "logsource": {"product": ls.get("product"),
                      "category": ls.get("category"),
                      "service": ls.get("service")},
        "mitre_attack_tags": tags[:4],
        "detection_selections": sel,
    }, ensure_ascii=False)
    return user, assistant


def make_split(ids: list[str], test_frac: float = 0.30, seed: int | None = None,
               group_of: dict | None = None):
    """
    Deterministic split. Commit the output next to the results.

    group_of given  -> ARTICLE-GROUPED split (Stage 1 fix, 26 Oct 2026):
                       all rules of one article go to the same side.
                       Default seed sk2_split.GROUPED_SPLIT_SEED.
    group_of None   -> the LEGACY per-rule split (seed 20260823), kept only
                       so the earlier results_* folders can be reproduced.
                       It leaks sibling rules across the split; do not use
                       it for new results.
    """
    if group_of is not None:
        from sk2_split import grouped_split, GROUPED_SPLIT_SEED
        return grouped_split(ids, group_of, test_frac=test_frac,
                             seed=GROUPED_SPLIT_SEED if seed is None else seed)
    seed = 20260823 if seed is None else seed
    rng = np.random.default_rng(seed)
    ids = sorted(ids)
    idx = rng.permutation(len(ids))
    n_test = max(1, int(round(len(ids) * test_frac)))
    test = sorted(ids[i] for i in idx[:n_test])
    train = sorted(ids[i] for i in idx[n_test:])
    return {"seed": seed, "test_frac": test_frac, "train": train, "test": test}


def save_split(split: dict, path: str | Path):
    Path(path).write_text(json.dumps(split, indent=2), encoding="utf-8")
