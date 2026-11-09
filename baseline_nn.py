r"""
baseline_nn.py  --  non-neural baselines on YOUR data.  $0. No API calls.

WHY THIS EXISTS
===============
"Run your data through another system to prove yours is better" is the right
instinct, but no comparable system can be run on this corpus: LLMCloudHunter is
AWS-CloudTrail-only, takes URLs rather than text, and ships no evaluator. The
standard answer in NLP when no competing system is runnable is to build the
baselines yourself, and to make them genuinely strong rather than straw men.

Three baselines, all free, all scored by the same evaluator on the same
instances as the main experiment:

  nn        NEAREST-NEIGHBOUR COPY. Retrieve the most similar training report
            by TF-IDF and emit its expert rule verbatim, unchanged. This is the
            baseline reviewers ask for in extraction papers, because it answers
            the sharpest question anyone can put to this work: does the language
            model actually read the report, or would copying a rule from a
            similar-looking report score just as well?

  majority  MOST FREQUENT LOG SOURCE. Always emit the single most common log
            source in the training split, with its most frequent fields and no
            values. A floor for routing and field selection.

  empty     EMPTY RULE. Emits nothing. The true zero, and the number that shows
            how much of any F1 comes from partial credit.

If SIGMA-KG does not beat `nn`, the retrieval component is doing the work and
the paper must say so. If it does beat it, that is a real, defensible claim of
the form your supervisor wants - measured on your data, against a baseline you
can describe in one sentence, with no vendor lock-in and nothing to install.

    python baseline_nn.py --bench data_bench --kg kg_out/sigma_kg.json \
        --scope 7 --out results_baseline

Writes per_instance.csv and summary.json in the same format as sk2_run.py, so
every downstream tool works on it unchanged.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sk2_kg import SigmaKG
from sk2_retrieval import ExemplarIndex, make_split
from sk2_evaluator import evaluate, aggregate

HEADLINE = [
    ("ls_exact",       None,            "Logsource exact"),
    ("ls_partial",     None,            "Logsource partial"),
    ("field_validity", None,            "Field validity (KG)"),
    ("field_f1",       "field_deg",     "Field names F1"),
    ("fieldmod_f1",    "fieldmod_deg",  "Field+modifier F1"),
    ("value_f1",       "value_deg",     "Values F1 (strict)"),
    ("value_sem_f1",   "value_sem_deg", "Values F1 (semantic)"),
    ("tech_f1",        "tech_deg",      "MITRE technique F1"),
    ("fieldvalue_f1",    "fieldvalue_deg",    "Field-value F1"),
    ("fieldmodvalue_f1", "fieldmodvalue_deg", "Field+mod-value F1"),
]


def strip_reference_fields(rule: dict) -> dict:
    """
    A retrieved training rule is a complete SigmaHQ rule. Emitting it verbatim
    would hand the baseline a title and description written for a DIFFERENT
    report, which is not what a system would output. We keep exactly the parts
    a generator produces: log source, detection, ATT&CK tags.
    """
    return {
        "title": "Retrieved rule (nearest neighbour)",
        "status": "experimental",
        "logsource": rule.get("logsource") or {},
        "detection": rule.get("detection") or {},
        "tags": [t for t in (rule.get("tags") or []) if str(t).startswith("attack.")],
        "level": rule.get("level", "medium"),
    }


def majority_rule(train_rules: list[dict], kg: SigmaKG, n_fields: int = 2) -> dict:
    """Most frequent log source in the training split, with its top fields."""
    c = collections.Counter()
    for r in train_rules:
        ls = r.get("logsource") or {}
        c[(ls.get("product"), ls.get("category"), ls.get("service"))] += 1
    (p, cat, s), _ = c.most_common(1)[0]
    fields = kg.allowed_fields(p, cat, s, top=n_fields)
    sel = {f: "__empty__" for f in fields} or {"__no_detection__": "__empty__"}
    return {"title": "Majority baseline", "status": "experimental",
            "logsource": {k: v for k, v in
                          (("product", p), ("category", cat), ("service", s)) if v},
            "detection": {"selection": sel, "condition": "selection"},
            "tags": [], "level": "medium"}


EMPTY_RULE = {"title": "Empty baseline", "status": "experimental",
              "logsource": {}, "tags": [], "level": "medium",
              "detection": {"selection": {"__no_detection__": "__empty__"},
                            "condition": "selection"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data_bench")
    ap.add_argument("--kg", default="kg_out/sigma_kg.json")
    ap.add_argument("--out", default="results_baseline")
    ap.add_argument("--strata", default="grounded,partial")
    ap.add_argument("--scope", type=int, default=7)
    ap.add_argument("--keep-grab-bag", action="store_true")
    ap.add_argument("--baselines", default="nn,majority,empty")
    ap.add_argument("--legacy-split", action="store_true",
                    help="OLD per-rule split, only to reproduce results_baseline")
    ap.add_argument("--split-seed", type=int, default=None)
    args = ap.parse_args()

    bench, out = Path(args.bench), Path(args.out)
    index = json.loads((bench / "index.json").read_text(encoding="utf-8"))
    strata = {s.strip() for s in args.strata.split(",")}
    index = [r for r in index if r["stratum"] in strata]
    ids = [r["id"] for r in index]
    stratum_of = {r["id"]: r["stratum"] for r in index}

    docs, rules, broken = {}, {}, 0
    for i in ids:
        a, r = bench / "articles" / f"{i}.txt", bench / "rules" / f"{i}.yml"
        try:
            docs[i] = a.read_text(encoding="utf-8", errors="ignore")
            rules[i] = yaml.safe_load(r.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            broken += 1
    ids = [i for i in ids if i in docs and i in rules]
    if broken:
        print(f"[*] {broken} pairs unreadable and skipped")

    # Stage 1 (26 Oct 2026): same article-grouped split as sk2_run.py, so the
    # baselines and the system are scored on identical instances.
    if args.legacy_split:
        group_of = None
        split = make_split(ids)
    else:
        from sk2_split import groups_from_bench
        group_of, _rep = groups_from_bench(bench, ids)
        split = make_split(ids, seed=args.split_seed, group_of=group_of)
    test_ids, train_ids = split["test"], split["train"]
    kg = SigmaKG(json_path=args.kg)
    print(f"[*] KG backend: {kg.status()['backend']}")

    scope = None
    if args.scope:
        scope = kg.scope_logsources(top=args.scope,
                                    exclude_grab_bag=not args.keep_grab_bag)
        want = {tuple(x or None for x in t) for t in scope}
        cov = kg.logsource_coverage(scope)
        print(f"[*] SCOPE: {len(scope)} log sources, {cov:.1%} of SigmaHQ")
        def _in_scope(rid):
            l = (rules[rid] or {}).get("logsource") or {}
            return (l.get("product"), l.get("category"), l.get("service")) in want

        before = len(test_ids)
        test_ids = [i for i in test_ids if _in_scope(i)]
        print(f"[*] test instances in scope: {len(test_ids)}/{before}")

    overlap = set(test_ids) & set(train_ids)
    if overlap:
        print(f"[!!] ABORT: {len(overlap)} instances in both splits.")
        return 2
    if group_of is not None:
        tg = {group_of[i] for i in train_ids}
        shared = [i for i in test_ids if group_of.get(i) in tg]
        if shared:
            print(f"[!!] ABORT: {len(shared)} test instances share an article "
                  f"with the retrieval pool.")
            return 2
    print(f"[*] pre-flight OK: {len(test_ids)} to score, "
          f"{len(train_ids)} in the retrieval pool, no overlap")

    exemplars = ExemplarIndex(docs, rules, train_ids, group_of=group_of)
    train_rules = [rules[i] for i in train_ids]
    maj = majority_rule(train_rules, kg)

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in [b.strip() for b in args.baselines.split(",")]:
        (out / "rules" / name).mkdir(parents=True, exist_ok=True)
        print(f"\n--- baseline '{name}' ---", flush=True)
        for rid in test_ids:
            if name == "nn":
                nb = exemplars.retrieve(rid, docs[rid], k=1)[0]
                pred = strip_reference_fields(nb["rule"])
            elif name == "majority":
                pred = maj
            else:
                pred = EMPTY_RULE
            m = evaluate(pred, rules[rid], kg)
            rows.append({"config": name, "id": rid, "stratum": stratum_of[rid], **m})
            (out / "rules" / name / f"{rid}.yml").write_text(
                yaml.safe_dump(pred, sort_keys=False, allow_unicode=True),
                encoding="utf-8")

    with (out / "per_instance.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)

    names = [b.strip() for b in args.baselines.split(",")]
    summary = {"model": "none (non-neural baselines)", "kg": kg.status(),
               "split_seed": split["seed"], "n_test": len(test_ids),
               "strata": sorted(strata), "scope_n": args.scope or None,
               "scope_coverage": round(kg.logsource_coverage(scope), 4) if scope else None,
               "configs": {}}

    print("\n" + "=" * 88)
    print(f"NON-NEURAL BASELINES   n={len(test_ids)}   (no API calls, $0)")
    print("=" * 88)
    print(f"{'metric':<26}" + "".join(f"{n:>18}" for n in names))
    print("-" * 88)
    for metric, deg, label in HEADLINE:
        line = f"{label:<26}"
        for n in names:
            sub = [r for r in rows if r["config"] == n]
            mval, lo, hi, k = aggregate(sub, metric, deg)
            summary["configs"].setdefault(n, {})[metric] = {
                "mean": mval, "ci_low": lo, "ci_high": hi, "n": k}
            line += f"{mval:>18.4f}"
        print(line)
    print("-" * 88)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nHOW TO READ THIS")
    print("  nn is the baseline that matters. It copies the expert rule of the")
    print("  most similar TRAINING report, unchanged, having never read the test")
    print("  report at all. Compare it to your `full` column:")
    print("    * field-level scores close to yours -> field selection is largely")
    print("      predictable from the genre, and you must say so.")
    print("    * value-level scores far below yours -> your system is genuinely")
    print("      reading the report, and that is the claim to make.")
    print(f"\n[+] {out/'summary.json'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
