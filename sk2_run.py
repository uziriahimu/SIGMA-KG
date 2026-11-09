"""
sk2_run.py  --  TASK 5, part 2

Runs the benchmark and produces everything a reviewer asked for:
per-instance CSV, bootstrap CIs, paired significance tests on REAL data,
per-stratum breakdown, and an ablation grid where each switch targets one
specific error class.

Configurations
--------------
    full            KG routing + KG field vocabulary + retrieved exemplars
    no_kg           no routing, no vocabulary constraint  -> logsource and field
                                                             validity should fall
    no_retrieval    fixed generic prompt, KG still on     -> field/value should fall
    no_both         neither                               -> the floor

None of them touch the reference rule. Ground truth is used for scoring only.

    python sk2_run.py --bench data_bench --kg kg_out/sigma_kg.json \
                      --configs full,no_kg,no_retrieval,no_both \
                      --strata grounded,partial --limit 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sk2_kg import SigmaKG
from sk2_retrieval import ExemplarIndex, make_split, save_split
from sk2_extractor import SchemaGroundedExtractor
from sk2_evaluator import (evaluate, aggregate, paired_test, macro_routing_f1,
                           conditional_metrics, hallucination_rates,
                           deployable_rate, error_profile)

CONFIGS = {
    "full":         dict(use_kg=True,  use_retrieval=True),
    "no_kg":        dict(use_kg=False, use_retrieval=True),
    "no_retrieval": dict(use_kg=True,  use_retrieval=False),
    "no_both":      dict(use_kg=False, use_retrieval=False),
    # ------------------------------------------------------------------
    # schema_prompt -- the baseline a reviewer will ask for.
    #
    # The model is TOLD every legal field name for every in-scope log source
    # (64 names, ~200 tokens for the seven-source scope) and nothing checks
    # that it complied. Everything else matches `full`: same exemplars, same
    # log-source menu, same routing.
    #
    # This isolates ENFORCEMENT from INFORMATION. If it lands near zero
    # invalid fields, the symbolic layer is redundant and the paper must say
    # so. If it reduces but does not eliminate them, that is the argument for
    # the graph: prompting shifts a distribution, enforcement gives a
    # guarantee. Either outcome is publishable; not running it is not.
    # ------------------------------------------------------------------
    "schema_prompt": dict(use_kg=True, use_retrieval=True,
                          schema_in_prompt=True, enforce_fields=False),
}

HEADLINE = [
    ("ls_exact",      None,             "Logsource exact"),
    ("ls_partial",    None,             "Logsource partial"),
    ("field_validity", None,            "Field validity (KG)"),
    ("field_f1",      "field_deg",      "Field names F1"),
    ("fieldmod_f1",   "fieldmod_deg",   "Field+modifier F1"),
    ("value_f1",      "value_deg",      "Values F1 (strict)"),
    ("value_sem_f1",  "value_sem_deg",  "Values F1 (semantic)"),
    ("tech_f1",       "tech_deg",       "MITRE technique F1"),
    # Stage 1 (26 Oct 2026): value counted only on the SAME field / field+modifier
    ("fieldvalue_f1",    "fieldvalue_deg",    "Field-value F1"),
    ("fieldmodvalue_f1", "fieldmodvalue_deg", "Field+mod-value F1"),
    ("abstained",     None,             "Abstention rate"),
]


def assemble(rec: dict, kg) -> dict:
    ls = rec.get("routed_logsource") or {}
    logsource = {k: v for k, v in ls.items() if v}
    if not logsource:
        logsource = {"product": "windows", "category": "process_creation"}
    sel = rec.get("selections") or {}
    if not sel:
        sel = {"__no_detection__": "__empty__"}
    return {
        "title": "Generated detection rule",
        "status": "experimental",
        "logsource": logsource,
        "detection": {"selection": sel, "condition": "selection"},
        "tags": rec.get("tags") or [],
        "level": "medium",
    }


def compiles(rule: dict, backend: str = "splunk") -> bool:
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
            yaml.safe_dump(rule, f, sort_keys=False, allow_unicode=True)
            path = f.name
        # --without-pipeline is required by sigma-cli 3.x; without it every
        # conversion is refused before the rule is even parsed, and you measure
        # 0% for every configuration - a tooling artefact, not a result.
        r = subprocess.run(["sigma", "convert", "--target", backend,
                            "--without-pipeline", str(path)],
                           capture_output=True, text=True, timeout=60,
                           encoding="utf-8", errors="replace")
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data_bench")
    ap.add_argument("--kg", default="kg_out/sigma_kg.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--configs", default="full,no_kg,no_retrieval,no_both")
    ap.add_argument("--strata", default="grounded,partial")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-on", choices=["test", "dev"], default="test",
                    help="'dev' scores a development set carved out of the "
                         "TRAINING pool, leaving the held-out test set untouched. "
                         "Tune on dev; report test once.")
    ap.add_argument("--dev-frac", type=float, default=0.20,
                    help="fraction of the training pool used as dev (default 0.20)")
    ap.add_argument("--test-limit", type=int, default=0,
                    help="score only the first N test instances. Exemplars and the "
                         "split are unchanged, so results stay paired with a full "
                         "run of the same seed. Use this to price an expensive model.")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--compile-check", action="store_true",
                    help="run sigma-cli on every generated rule (slow)")
    ap.add_argument("--scope", type=int, default=0,
                    help="restrict the benchmark to the N most frequent log sources "
                         "(0 = all). --scope 7 covers 62.5%% of SigmaHQ with a median "
                         "of 9 fields, comparable to LLMCloudHunter's single 23-field "
                         "log source. The scope is reported, never hidden.")
    ap.add_argument("--keep-grab-bag", action="store_true",
                    help="keep windows/security in scope (91 fields, a hundred "
                         "unrelated event types under one name)")
    ap.add_argument("--menu-top", type=int, default=120,
                    help="how many log sources the KG offers the model when not "
                         "scoped. Must cover the label space or the constrained "
                         "config is handicapped against the unconstrained one.")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible endpoint for an open-source model, e.g. "
                         "http://localhost:11434/v1 (Ollama) or "
                         "https://openrouter.ai/api/v1. Same pipeline, different model.")
    ap.add_argument("--neo4j-uri", default=None)
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-pass", default=None)
    # Stage 1 (26 Oct 2026)
    ap.add_argument("--legacy-split", action="store_true",
                    help="use the OLD per-rule split (seed 20260823) only to "
                         "reproduce earlier results_* folders. It leaks sibling "
                         "rules of one article across the split.")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="seed for the article-grouped split "
                         "(default sk2_split.GROUPED_SPLIT_SEED = 20261026)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the split, run every pre-flight check, write "
                         "split.json, and stop BEFORE any API call. Costs $0.")
    args = ap.parse_args()

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("[!] OPENAI_API_KEY not set"); return 2

    bench = Path(args.bench)
    index = json.loads((bench / "index.json").read_text(encoding="utf-8"))
    strata = set(s.strip() for s in args.strata.split(","))
    index = [r for r in index if r["stratum"] in strata]
    if args.limit:
        index = index[: args.limit]
    ids = [r["id"] for r in index]
    stratum_of = {r["id"]: r["stratum"] for r in index}
    print(f"[*] {len(ids)} instances in strata {sorted(strata)}")

    # Load the pairs defensively. An interrupted build can leave index.json
    # naming a file that was never written; that used to crash the whole run
    # on the first bad id. Skip and count instead, so one damaged pair cannot
    # cost you an entire experiment.
    docs, rules, broken = {}, {}, []
    for i in ids:
        a = bench / "articles" / f"{i}.txt"
        r = bench / "rules" / f"{i}.yml"
        try:
            text = a.read_text(encoding="utf-8", errors="ignore")
            rule = yaml.safe_load(r.read_text(encoding="utf-8"))
        except Exception as e:
            broken.append((i, type(e).__name__))
            continue
        if not text.strip() or not isinstance(rule, dict):
            broken.append((i, "empty"))
            continue
        docs[i], rules[i] = text, rule

    if broken:
        print(f"[!] {len(broken)} of {len(ids)} pairs are missing or unreadable "
              f"and were skipped, e.g. {[b[0] for b in broken[:5]]}")
        print("    Your benchmark is damaged - most likely an interrupted build.")
        print("    Rebuild it with 2_BUILD_BENCHMARK.bat to get clean data.")
        ids = [i for i in ids if i in docs]
        stratum_of = {i: stratum_of[i] for i in ids}

    if len(ids) < 10:
        print(f"[!] only {len(ids)} usable pairs left - not enough to run. Rebuild first.")
        return 2

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.legacy_split:
        group_of = None
        split = make_split(ids)
        print("[!] LEGACY per-rule split: sibling rules of one article can sit "
              "on both sides. Use only to reproduce old results.")
    else:
        from sk2_split import groups_from_bench
        group_of, grep_ = groups_from_bench(bench, ids)
        split = make_split(ids, seed=args.split_seed, group_of=group_of)
        split["group_report"] = grep_
        split["group_of"] = group_of
        print(f"[*] ARTICLE-GROUPED split: {grep_['n_ids']} pairs in "
              f"{grep_['n_groups']} articles ({grep_['n_multi_rule_groups']} with "
              f"2+ rules; merged by url {grep_['by_url']}, text {grep_['by_text']}, "
              f"near-dup {grep_['by_near_dup']})")
    save_split(split, out / "split.json")

    # ------------------------------------------------------------------
    # --eval-on dev is the honest way to try ideas.
    #
    # If you tune a setting by looking at the TEST score, the test score
    # stops being a fair estimate - you fitted it. So carve a development
    # set out of the TRAINING pool instead, tune against that as often as
    # you like, then run --eval-on test exactly once with the winner.
    #
    # The exemplar index is built from the training pool MINUS the dev set,
    # so a dev report can never retrieve itself as its own example.
    # ------------------------------------------------------------------
    if args.eval_on == "dev" and group_of is not None:
        from sk2_split import carve_dev
        test_ids, pool_ids = carve_dev(split["train"], group_of,
                                       args.dev_frac, split["seed"] + 1)
        (out / "dev_split.json").write_text(
            json.dumps({"seed": split["seed"] + 1, "grouped_by": "article",
                        "dev": test_ids, "exemplar_pool": pool_ids}, indent=2),
            encoding="utf-8")
        print(f"[*] EVAL-ON=DEV  dev {len(test_ids)}  exemplar pool {len(pool_ids)}"
              f"  (grouped by article)")
        print( "[*] the held-out test set is NOT touched by this run")
    elif args.eval_on == "dev":
        tr = sorted(split["train"])
        order = list(range(len(tr)))
        random.Random(split["seed"] + 1).shuffle(order)
        n_dev = max(10, int(round(len(tr) * args.dev_frac)))
        test_ids = sorted(tr[i] for i in order[:n_dev])
        pool_ids = sorted(tr[i] for i in order[n_dev:])
        (out / "dev_split.json").write_text(
            json.dumps({"seed": split["seed"] + 1, "dev": test_ids,
                        "exemplar_pool": pool_ids}, indent=2), encoding="utf-8")
        print(f"[*] EVAL-ON=DEV  dev {len(test_ids)}  exemplar pool {len(pool_ids)}")
        print( "[*] the held-out test set is NOT touched by this run")
    else:
        test_ids = split["test"]
        pool_ids = split["train"]
        print(f"[*] EVAL-ON=TEST  train {len(pool_ids)}  test {len(test_ids)}  "
              f"(seed {split['seed']})")

    neo = None
    if args.neo4j_uri:
        # Prefer the environment. Passing a password as a command-line
        # argument writes it into your shell history and into the process
        # list; reading NEO4J_PASSWORD from .env does neither.
        neo = {"enabled": True, "uri": args.neo4j_uri,
               "user": args.neo4j_user,
               "password": args.neo4j_pass or os.environ.get("NEO4J_PASSWORD") or ""}
    kg = SigmaKG(json_path=args.kg, neo4j_cfg=neo)
    print(f"[*] KG backend: {kg.status()}")

    scope = None
    if args.scope:
        scope = kg.scope_logsources(top=args.scope,
                                    exclude_grab_bag=not args.keep_grab_bag)
        cov = kg.logsource_coverage(scope)
        print(f"[*] SCOPE: {len(scope)} log sources, {cov:.1%} of all SigmaHQ rules")
        for t in scope:
            print(f"      {'/'.join(str(x or '-') for x in t):32} "
                  f"{len(kg.allowed_fields(*t))} fields")
        want = {tuple(x or None for x in t) for t in scope}

        def _in_scope(rid):
            l = (rules[rid] or {}).get("logsource") or {}
            return (l.get("product"), l.get("category"), l.get("service")) in want

        before = len(test_ids)
        test_ids = [i for i in test_ids if _in_scope(i)]
        print(f"[*] test instances in scope: {len(test_ids)}/{before}")
        if len(test_ids) < 10:
            print("[!] too few in-scope test instances - widen --scope or rebuild bigger")
            return 2

    # ------------------------------------------------------------------
    # --test-limit truncates ONLY the test set. The training pool, and so
    # the exemplar index, is untouched. That matters: --limit shrinks the
    # pool before the split, which quietly weakens retrieval and makes the
    # run incomparable to the full one. --test-limit takes a strict PREFIX
    # of the same seeded test order, so an expensive model can be scored on
    # a subset that a cheap model already covered, and the two are paired
    # on identical instances.
    # ------------------------------------------------------------------
    if args.test_limit and args.test_limit < len(test_ids):
        print(f"[*] TEST-LIMIT: scoring the first {args.test_limit} of "
              f"{len(test_ids)} in-scope instances (same seeded order, so this "
              f"is a paired subset of the full run)")
        test_ids = test_ids[: args.test_limit]

    # ------------------------------------------------------------------
    # PRE-FLIGHT. Everything below this line costs money, so verify the
    # things that would make the run worthless BEFORE the first API call.
    # ------------------------------------------------------------------
    overlap = set(test_ids) & set(pool_ids)
    if overlap:
        print(f"[!!] ABORT: {len(overlap)} reports are in BOTH the evaluation set "
              f"and the exemplar pool, e.g. {sorted(overlap)[:3]}. "
              f"That is ground-truth leakage. Nothing was spent.")
        return 2
    if args.eval_on == "dev" and (set(test_ids) & set(split["test"])):
        print("[!!] ABORT: the dev set overlaps the held-out test set. "
              "Nothing was spent.")
        return 2
    if group_of is not None:
        pool_groups = {group_of.get(j) for j in pool_ids}
        shared_art = sorted(i for i in test_ids if group_of.get(i) in pool_groups)
        if shared_art:
            print(f"[!!] ABORT: {len(shared_art)} evaluation reports share an ARTICLE "
                  f"with the exemplar pool, e.g. {shared_art[:3]}. Nothing was spent.")
            return 2
    missing = [i for i in test_ids if i not in docs or i not in rules]
    if missing:
        print(f"[!!] ABORT: {len(missing)} evaluation reports have no document or "
              f"no reference rule, e.g. {missing[:3]}. Nothing was spent.")
        return 2
    print(f"[*] pre-flight OK: {len(test_ids)} to score, "
          f"{len(pool_ids)} in the exemplar pool, no overlap"
          + (" (ids and articles)" if group_of is not None else " (ids only)"))
    if args.dry_run:
        print(f"[*] DRY RUN: split written to {out / 'split.json'}. "
              f"No API call was made.")
        return 0

    exemplars = ExemplarIndex(docs, rules, pool_ids, group_of=group_of)

    rows, meta = [], {}
    for cname in [c.strip() for c in args.configs.split(",")]:
        cfg = CONFIGS[cname]
        print(f"\n--- {cname}  {cfg} ---", flush=True)
        ex = SchemaGroundedExtractor(kg=kg, index=exemplars, model_name=args.model,
                                     k_exemplars=args.k, scope=scope,
                                     base_url=args.base_url,
                                     menu_top=args.menu_top, **cfg)
        t0 = time.time()
        for n, rid in enumerate(test_ids, 1):
            rec = ex.generate(docs[rid], report_id=rid)
            rule = assemble(rec, kg)
            m = evaluate(rule, rules[rid], kg)
            prop = rec.get("proposed_logsource") or {}
            row = {"config": cname, "id": rid, "stratum": stratum_of[rid],
                   "generation_failed": rec.get("generation_failed"),
                   "n_allowed_fields": rec.get("n_allowed_fields"),
                   "n_dropped_fields": rec.get("n_dropped_fields"),
                   "n_empty_fields": rec.get("n_empty_fields"),
                   "n_ungrounded_values": len(rec.get("ungrounded_values") or []),
                   "n_observables": rec.get("n_observables"),
                   # how often did the model name a log source that does not
                   # exist, and the graph had to correct it? a reportable number
                   "logsource_snapped": 1.0 if rec.get("logsource_snapped") else 0.0,
                   "proposed_logsource": "/".join(
                       str(prop.get(k) or "-") for k in ("product", "category", "service")),
                   "pred_logsource": "/".join(
                       str((rule.get("logsource") or {}).get(k) or "-")
                       for k in ("product", "category", "service")),
                   "gt_logsource": "/".join(
                       str((rules[rid].get("logsource") or {}).get(k) or "-")
                       for k in ("product", "category", "service")),
                   # did retrieval actually happen? if a config claims to use
                   # exemplars but this is 0, the run is not what it says it is
                   "retrieval_used": 1.0 if rec.get("used_retrieval") else 0.0,
                   "n_exemplars": len(rec.get("exemplars") or []),
                   "dropped_fields": ";;".join(rec.get("dropped_fields") or []),
                   **m}
            if args.compile_check:
                row["compiles_splunk"] = 1.0 if compiles(rule) else 0.0
            rows.append(row)
            (out / "rules" / cname).mkdir(parents=True, exist_ok=True)
            (out / "rules" / cname / f"{rid}.yml").write_text(
                yaml.safe_dump(rule, sort_keys=False, allow_unicode=True), encoding="utf-8")
            if n % 25 == 0:
                print(f"    {n}/{len(test_ids)}  ({time.time()-t0:.0f}s)", flush=True)
        meta[cname] = {"n": len(test_ids), "config": cfg}

        # integrity check per configuration - catches a run that silently
        # became a different experiment than the one you are paying for
        sub = [r for r in rows if r["config"] == cname]
        failed = sum(1 for r in sub if r.get("generation_failed"))
        if failed:
            print(f"    [!] {failed}/{len(sub)} generations FAILED (API errors). "
                  f"Treat this configuration as unreliable.")
        if cfg.get("use_retrieval"):
            no_ret = sum(1 for r in sub if not r.get("retrieval_used"))
            if no_ret:
                print(f"    [!] {no_ret}/{len(sub)} reports got NO exemplars although "
                      f"'{cname}' is supposed to use them. This configuration is "
                      f"not what its name says - do not report it.")

    csv_path = out / "per_instance.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)

    configs = [c.strip() for c in args.configs.split(",")]
    summary = {"model": args.model, "base_url": args.base_url, "kg": kg.status(), "split_seed": split["seed"],
               "split_grouped_by": split.get("grouped_by", "rule (legacy)"),
               "same_article_exemplars_skipped": exemplars.same_article_skipped,
               "eval_on": args.eval_on, "dev_frac": args.dev_frac if args.eval_on == "dev" else None,
               "test_limit": args.test_limit or None,
               "n_test": len(test_ids), "strata": sorted(strata),
               "scope_n": args.scope or None,
               "scope": ["/".join(str(x or "-") for x in t) for t in scope] if scope else None,
               "scope_coverage": round(kg.logsource_coverage(scope), 4) if scope else None,
               "configs": {}}

    print("\n" + "=" * 96)
    print(f"RESULTS   n={len(test_ids)}  set={args.eval_on.upper()}  "
          f"model={args.model}  KG={kg.status()['backend']}")
    if args.eval_on == "dev":
        print("          DEV NUMBERS - for choosing settings only. "
              "Do not put these in the paper.")
    print("=" * 96)
    print(f"{'metric':<26}" + "".join(f"{c:>17}" for c in configs))
    print("-" * 96)
    for metric, deg, label in HEADLINE:
        line = f"{label:<26}"
        for c in configs:
            sub = [r for r in rows if r["config"] == c]
            mval, lo, hi, n = aggregate(sub, metric, deg)
            summary["configs"].setdefault(c, {})[metric] = {
                "mean": mval, "ci_low": lo, "ci_high": hi, "n": n}
            line += f"{mval:>17.4f}"
        print(line)
    if args.compile_check:
        line = f"{'Compiles (splunk)':<26}"
        for c in configs:
            sub = [r for r in rows if r["config"] == c]
            mval, lo, hi, n = aggregate(sub, "compiles_splunk")
            summary["configs"].setdefault(c, {})["compiles_splunk"] = {
                "mean": mval, "ci_low": lo, "ci_high": hi, "n": n}
            line += f"{mval:>17.4f}"
        print(line)
    print("-" * 96)

    # per-stratum
    print("\nPER STRATUM (value_sem_f1)")
    for c in configs:
        for s in sorted(strata):
            sub = [r for r in rows if r["config"] == c and r["stratum"] == s]
            if sub:
                mval, lo, hi, n = aggregate(sub, "value_sem_f1", "value_sem_deg")
                print(f"   {c:<14} {s:<11} {mval:.4f}  [{lo:.4f}, {hi:.4f}]  n={n}")

    # ---- METRIC SUITE v2 : the blocks written for the reviewers ----
    summary["suite"] = {}
    for c in configs:
        sub = [r for r in rows if r["config"] == c]
        gtmap = {r["id"]: rules[r["id"]] for r in sub if r["id"] in rules}
        summary["suite"][c] = {
            "routing": macro_routing_f1(sub),
            "conditional": conditional_metrics(sub),
            "hallucination": hallucination_rates(sub),
            "deployable": deployable_rate(sub, require_compile=args.compile_check),
            "errors": error_profile([(r, gtmap.get(r["id"]), kg) for r in sub]),
        }

    print("\n" + "=" * 96)
    print("ROUTING  (accuracy is inflated by the majority class; macro-F1 is not)")
    print("=" * 96)
    print(f"{'config':<16}{'accuracy':>11}{'macro-F1':>11}{'majority':>11}{'classes':>10}")
    for c in configs:
        r = summary["suite"][c]["routing"]
        print(f"{c:<16}{r['accuracy']:>11.4f}{r['macro_f1']:>11.4f}"
              f"{r['majority_baseline']:>11.4f}{r['n_classes']:>10}")

    print("\n" + "=" * 96)
    print("CONDITIONAL  (isolates the graph's routing effect from its vocabulary effect)")
    print("=" * 96)
    print(f"{'config':<16}{'routed ok':>11}{'field|routed':>14}{'val_sem|routed':>16}{'val_sem|fields':>16}")
    for c in configs:
        d = summary["suite"][c]["conditional"]
        def f(x): return f"{x:.4f}" if isinstance(x, float) else "  -   "
        print(f"{c:<16}{d['n_routed_correct']:>11}{f(d.get('field_f1_given_routed')):>14}"
              f"{f(d.get('value_sem_f1_given_routed')):>16}"
              f"{f(d.get('value_sem_f1_given_fields')):>16}")

    print("\n" + "=" * 96)
    print("HALLUCINATION  (three kinds - none of these are reported by prior work)")
    print("=" * 96)
    print(f"{'config':<16}{'schema':>11}{'logsource':>12}{'value':>11}{'empty pad':>12}")
    for c in configs:
        h = summary["suite"][c]["hallucination"]
        def g(x): return f"{x:.4f}" if isinstance(x, float) else "  -   "
        print(f"{c:<16}{g(h['schema_hallucination_rate']):>11}"
              f"{g(h['logsource_hallucination_rate']):>12}"
              f"{g(h['value_hallucination_rate']):>11}{h['empty_field_rate']:>12.2f}")

    print("\n" + "=" * 96)
    print("ERROR ATTRIBUTION  (reviewer FMrU: the boundary between neural and symbolic)")
    print("=" * 96)
    for c in configs:
        e = summary["suite"][c]["errors"]
        print(f"  {c}:  neural {e['neural_share']:.1%}   symbolic {e['symbolic_share']:.1%}")
        for k, v in sorted(e["counts"].items(), key=lambda kv: -kv[1]):
            print(f"        {k:<22} {v:>4}   {e['share'][k]:.1%}")

    if args.compile_check:
        print("\nDEPLOYABLE RATE  (compiles AND valid fields AND correct log source)")
        for c in configs:
            d = summary["suite"][c]["deployable"]
            print(f"  {c:<16} {d['deployable_rate']}   (n={d['n_scored']})")

    # ablations, paired on real per-instance data
    print("\nPAIRED TESTS vs full  (Wilcoxon, real per-instance scores)")
    print("-" * 96)
    base = {r["id"]: r for r in rows if r["config"] == "full"}
    summary["paired"] = {}
    for c in configs:
        if c == "full":
            continue
        other = {r["id"]: r for r in rows if r["config"] == c}
        common = sorted(set(base) & set(other))
        for metric, deg, label in HEADLINE:
            a = [base[i][metric] for i in common if base[i].get(metric) is not None
                 and other[i].get(metric) is not None]
            b = [other[i][metric] for i in common if base[i].get(metric) is not None
                 and other[i].get(metric) is not None]
            if len(a) < 5:
                continue
            t = paired_test(a, b)
            summary["paired"][f"full_vs_{c}|{metric}"] = t
            flag = "SIG" if t["p"] < 0.05 else "ns "
            print(f"   full vs {c:<13} {label:<24} diff={t['mean_diff']:+.4f} "
                  f"p={t['p']:.4f} {flag}  W/L/T={t['wins']}/{t['losses']}/{t['ties']}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[+] {csv_path}")
    print(f"[+] {out/'summary.json'}")
    print(f"[+] {out/'split.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
