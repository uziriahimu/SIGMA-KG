"""
sigma_kg_miner.py  --  TASK 2, part 1

Mines the real Sigma field vocabulary out of SigmaHQ and emits a knowledge
graph that Neo4j can load.

Why this exists
---------------
Your current Neo4j holds MITRE ATT&CK STIX only. It knows nothing about Sigma
fields, so `graph_engine.py` can only ever touch `tags`. This script produces
the missing half: which detection fields legitimately occur under which log
source, and which modifiers each field accepts.

Once loaded, the pipeline can:
    * route a report to a log source        (Technique -> DataSource -> LogSource)
    * restrict the LLM to that log source's real field list   -> validity 100%
    * pick a modifier the field actually accepts

Outputs (into --out):
    sigma_kg.json      full graph, human-readable
    sigma_kg.cypher    idempotent Neo4j loader (MERGE, safe to re-run)
    kg_report.txt      coverage summary for the paper

Usage
-----
    # mine the rules already on disk
    python sigma_kg_miner.py --rules-dir data_sigma_bench/gt_sigma_rules --out kg_out

    # or mine ALL of SigmaHQ (needs internet, ~3000 rules, a few minutes)
    python sigma_kg_miner.py --download --out kg_out
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

BLOCK_SKIP = {"condition", "timeframe"}

# Sigma value modifiers per the specification. Anything outside this set that
# appears in the corpus is recorded too, but flagged in the report.
KNOWN_MODIFIERS = {
    "contains", "startswith", "endswith", "all", "re", "base64", "base64offset",
    "utf16", "utf16le", "utf16be", "wide", "windash", "cidr", "lt", "lte",
    "gt", "gte", "expand", "cased", "exists", "fieldref",
}


# ----------------------------------------------------------------------
# Rule walking
# ----------------------------------------------------------------------

def iter_field_specs(detection):
    """Yield (raw_key, value) for every field spec inside every detection block."""
    if not isinstance(detection, dict):
        return

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield str(k), v
                yield from walk(v)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)

    for block_name, block in detection.items():
        if str(block_name).lower() in BLOCK_SKIP:
            continue
        # `keywords` blocks are bare value lists, not field specs
        if str(block_name).lower().startswith("keyword"):
            continue
        yield from walk(block)


def split_key(raw_key: str):
    """'Image|endswith|all' -> ('Image', ['endswith', 'all'])"""
    parts = str(raw_key).split("|")
    return parts[0].strip(), [p.strip().lower() for p in parts[1:] if p.strip()]


def logsource_keys(ls: dict):
    """
    A rule's logsource may specify any subset of product/category/service.
    Index it under every useful lookup key so routing can fall back gracefully.
    Returns a list of (kind, key) pairs, most specific first.
    """
    p = (ls or {}).get("product")
    c = (ls or {}).get("category")
    s = (ls or {}).get("service")
    p = str(p).strip().lower() if p else None
    c = str(c).strip().lower() if c else None
    s = str(s).strip().lower() if s else None

    keys = []
    if p and c and s: keys.append(("pcs", f"{p}|{c}|{s}"))
    if p and c:       keys.append(("pc",  f"{p}|{c}"))
    if p and s:       keys.append(("ps",  f"{p}|{s}"))
    if c and s:       keys.append(("cs",  f"{c}|{s}"))
    if p:             keys.append(("p",   p))
    if c:             keys.append(("c",   c))
    if s:             keys.append(("s",   s))
    return keys, (p, c, s)


# ----------------------------------------------------------------------
# Mining
# ----------------------------------------------------------------------

def mine(rule_paths):
    fields_by_key = defaultdict(Counter)          # logsource key -> field -> count
    mods_by_field = defaultdict(Counter)          # "logsourcekey::field" -> modifier -> count
    value_shape = defaultdict(Counter)            # "logsourcekey::field" -> shape -> count
    triples = Counter()                           # (product, category, service) -> count
    tech_to_logsource = defaultdict(Counter)      # attack technique -> logsource key
    cond_shapes = Counter()
    stats = Counter()
    unknown_mods = Counter()

    for path in rule_paths:
        try:
            docs = list(yaml.safe_load_all(Path(path).read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            stats["unparseable"] += 1
            continue

        for rule in docs:
            if not isinstance(rule, dict) or "detection" not in rule:
                continue
            stats["rules"] += 1

            ls = rule.get("logsource") or {}
            keys, triple = logsource_keys(ls)
            if not keys:
                stats["no_logsource"] += 1
                continue
            triples[triple] += 1

            det = rule.get("detection") or {}
            cond = str(det.get("condition", "")).strip()
            cond_shapes["selection" if cond == "selection" else "complex"] += 1

            tags = [str(t).lower() for t in (rule.get("tags") or [])]
            techs = [t.replace("attack.", "") for t in tags
                     if re.match(r"^attack\.t\d{4}(\.\d{3})?$", t)]

            seen_in_rule = set()
            for raw_key, val in iter_field_specs(det):
                field, mods = split_key(raw_key)
                if not field or field.lower() in BLOCK_SKIP:
                    continue
                # block names nested one level down are not fields
                if isinstance(val, (dict,)) and not mods:
                    continue
                stats["field_specs"] += 1
                shape = classify_value(val)

                for _kind, key in keys:
                    fields_by_key[key][field] += 1
                    fq = f"{key}::{field}"
                    for m in mods:
                        mods_by_field[fq][m] += 1
                        if m not in KNOWN_MODIFIERS:
                            unknown_mods[m] += 1
                    if not mods:
                        mods_by_field[fq]["<none>"] += 1
                    value_shape[fq][shape] += 1
                seen_in_rule.add(field)

            for t in techs:
                for _kind, key in keys:
                    tech_to_logsource[t][key] += 1

    return dict(
        fields_by_key=fields_by_key, mods_by_field=mods_by_field,
        value_shape=value_shape, triples=triples,
        tech_to_logsource=tech_to_logsource, cond_shapes=cond_shapes,
        stats=stats, unknown_mods=unknown_mods,
    )


def classify_value(v):
    """Coarse shape of a literal, used later to choose a modifier."""
    if isinstance(v, bool):   return "bool"
    if isinstance(v, int):    return "int"
    if isinstance(v, list):   return "list"
    if v is None:             return "null"
    s = str(v)
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}(/\d{1,2})?", s):      return "ip"
    if re.fullmatch(r"[0-9a-fA-F]{32,64}", s):                    return "hash"
    if s.startswith("http://") or s.startswith("https://"):       return "url"
    if s.startswith("\\") or s.startswith("/") or ":\\" in s:     return "path"
    if "*" in s:                                                  return "wildcard"
    return "token"


# ----------------------------------------------------------------------
# Emit
# ----------------------------------------------------------------------

def build_graph(m, min_count: int = 1):
    fields_by_key = m["fields_by_key"]
    mods_by_field = m["mods_by_field"]
    value_shape = m["value_shape"]

    logsources = []
    for key, counter in sorted(fields_by_key.items()):
        total = sum(counter.values())
        fields = []
        for fname, cnt in counter.most_common():
            if cnt < min_count:
                continue
            fq = f"{key}::{fname}"
            fields.append({
                "name": fname,
                "count": cnt,
                "share": round(cnt / total, 4),
                "modifiers": dict(mods_by_field[fq].most_common()),
                "value_shapes": dict(value_shape[fq].most_common()),
            })
        logsources.append({"key": key, "n_field_specs": total, "fields": fields})

    return {
        "meta": {
            "rules_mined": m["stats"]["rules"],
            "field_specs": m["stats"]["field_specs"],
            "unparseable": m["stats"]["unparseable"],
            "rules_without_logsource": m["stats"]["no_logsource"],
            "condition_shapes": dict(m["cond_shapes"]),
            "modifiers_outside_spec": dict(m["unknown_mods"]),
        },
        "logsource_triples": [
            {"product": p, "category": c, "service": s, "count": n}
            for (p, c, s), n in m["triples"].most_common()
        ],
        "logsources": logsources,
        "technique_to_logsource": {
            t: dict(c.most_common(5)) for t, c in m["tech_to_logsource"].items()
        },
    }


def esc(s):
    return str(s).replace("\\", "\\\\").replace("'", "\\'")


def emit_cypher(graph) -> str:
    L = [
        "// Sigma field-vocabulary knowledge graph",
        "// Idempotent: MERGE only. Safe to re-run.",
        "CREATE CONSTRAINT sigma_logsource_key IF NOT EXISTS",
        "  FOR (l:SigmaLogSource) REQUIRE l.key IS UNIQUE;",
        "CREATE CONSTRAINT sigma_field_key IF NOT EXISTS",
        "  FOR (f:SigmaField) REQUIRE f.fq IS UNIQUE;",
        "",
    ]
    for ls in graph["logsources"]:
        L.append(f"MERGE (l:SigmaLogSource {{key:'{esc(ls['key'])}'}}) "
                 f"SET l.n_field_specs={ls['n_field_specs']};")
        for f in ls["fields"]:
            fq = f"{ls['key']}::{f['name']}"
            mods = [m for m in f["modifiers"] if m != "<none>"]
            shapes = list(f["value_shapes"].keys())
            L.append(
                f"MERGE (f:SigmaField {{fq:'{esc(fq)}'}}) "
                f"SET f.name='{esc(f['name'])}', f.count={f['count']}, f.share={f['share']}, "
                f"f.modifiers={json.dumps(mods)}, f.value_shapes={json.dumps(shapes)};"
            )
            L.append(
                f"MATCH (l:SigmaLogSource {{key:'{esc(ls['key'])}'}}), "
                f"(f:SigmaField {{fq:'{esc(fq)}'}}) "
                f"MERGE (l)-[:HAS_FIELD {{count:{f['count']}}}]->(f);"
            )
    for tech, keys in graph["technique_to_logsource"].items():
        L.append(f"MERGE (t:AttackTechnique {{id:'{esc(tech)}'}});")
        for key, n in keys.items():
            L.append(
                f"MATCH (t:AttackTechnique {{id:'{esc(tech)}'}}), "
                f"(l:SigmaLogSource {{key:'{esc(key)}'}}) "
                f"MERGE (t)-[:OBSERVED_IN {{count:{n}}}]->(l);"
            )
    return "\n".join(L) + "\n"


def emit_report(graph) -> str:
    md = graph["meta"]
    out = []
    A = out.append
    A("=" * 74)
    A("SIGMA FIELD-VOCABULARY KNOWLEDGE GRAPH -- COVERAGE REPORT")
    A("=" * 74)
    A(f"rules mined                : {md['rules_mined']}")
    A(f"field specifications       : {md['field_specs']}")
    A(f"unparseable files          : {md['unparseable']}")
    A(f"rules without a logsource  : {md['rules_without_logsource']}")
    A(f"condition shapes           : {md['condition_shapes']}")
    A(f"logsource nodes            : {len(graph['logsources'])}")
    A(f"distinct (product,category,service) triples : {len(graph['logsource_triples'])}")
    A(f"techniques mapped to a logsource            : {len(graph['technique_to_logsource'])}")
    if md["modifiers_outside_spec"]:
        A(f"!! modifiers outside the Sigma spec        : {md['modifiers_outside_spec']}")
    A("")
    A("TOP LOGSOURCES BY FIELD-SPEC VOLUME")
    A("-" * 74)
    for ls in sorted(graph["logsources"], key=lambda x: -x["n_field_specs"])[:15]:
        names = ", ".join(f["name"] for f in ls["fields"][:8])
        A(f"  {ls['key']:<38} {ls['n_field_specs']:>5} specs  {len(ls['fields']):>3} fields")
        A(f"      {names}")
    A("")
    A("VOCABULARY SIZE PER LOGSOURCE  (this is what constrains the LLM)")
    A("-" * 74)
    sizes = sorted((len(l["fields"]) for l in graph["logsources"]), reverse=True)
    if sizes:
        A(f"  max {sizes[0]}   median {sizes[len(sizes)//2]}   min {sizes[-1]}")
    return "\n".join(out)


# ----------------------------------------------------------------------

def download_sigmahq(dest: Path):
    """Clone SigmaHQ shallowly. Requires git and internet."""
    import subprocess
    dest.mkdir(parents=True, exist_ok=True)
    repo = dest / "sigma"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/SigmaHQ/sigma.git", str(repo)], check=True)
    return sorted((repo / "rules").rglob("*.yml"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-dir", type=str, default=None,
                    help="directory of Sigma .yml rules to mine")
    ap.add_argument("--download", action="store_true",
                    help="git clone SigmaHQ and mine every rule (~3000)")
    ap.add_argument("--out", type=str, default="kg_out")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--exclude-index", type=str, default=None,
                    help="path to data_bench/index.json. Every rule used in the "
                         "benchmark is REMOVED before mining, so the schema graph "
                         "is provably disjoint from the evaluation set. Use this "
                         "for the numbers that go in the paper.")
    args = ap.parse_args()

    paths = []
    if args.download:
        print("[*] cloning SigmaHQ ...")
        paths = download_sigmahq(Path(args.out) / "_sigmahq")
    if args.rules_dir:
        paths += sorted(Path(args.rules_dir).rglob("*.yml"))
        paths += sorted(Path(args.rules_dir).rglob("*.yaml"))
    if not paths:
        print("[!] nothing to mine. pass --rules-dir and/or --download")
        return 2

    # ------------------------------------------------------------------
    # Leakage guard.
    #
    # The graph stores which FIELD NAMES are legal under which log source.
    # If it is mined from the same rule files that later serve as ground
    # truth, a reviewer can say the schema saw the answers - and that is
    # exactly the objection that sank the previous paper. Removing the
    # benchmark rules costs almost nothing (field vocabulary is shared
    # across thousands of rules) and removes the objection completely.
    # ------------------------------------------------------------------
    if args.exclude_index:
        idx = json.loads(Path(args.exclude_index).read_text(encoding="utf-8"))
        banned_abs, banned_name = set(), set()
        for row in idx:
            src = row.get("source_rule")
            if not src:
                continue
            try:
                banned_abs.add(str(Path(src).resolve()).lower())
            except Exception:
                pass
            banned_name.add(Path(src).name.lower())

        before = len(paths)
        kept = []
        for p in paths:
            try:
                a = str(Path(p).resolve()).lower()
            except Exception:
                a = str(p).lower()
            if a in banned_abs or Path(p).name.lower() in banned_name:
                continue
            kept.append(p)
        paths = kept
        print(f"[*] leakage guard: {before - len(paths)} benchmark rules excluded "
              f"({len(idx)} listed in {args.exclude_index}); {len(paths)} remain")
        if before - len(paths) == 0:
            print("[!] WARNING: nothing was excluded. Check that --rules-dir points "
                  "at the same SigmaHQ clone the benchmark was built from.")

    print(f"[*] mining {len(paths)} rule files ...")
    m = mine(paths)
    graph = build_graph(m, min_count=args.min_count)
    graph["meta"]["leakage_guard"] = bool(args.exclude_index)
    graph["meta"]["excluded_benchmark_rules"] = (
        before - len(paths) if args.exclude_index else 0)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "sigma_kg.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    (out / "sigma_kg.cypher").write_text(emit_cypher(graph), encoding="utf-8")
    report = emit_report(graph)
    (out / "kg_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[+] {out/'sigma_kg.json'}")
    print(f"[+] {out/'sigma_kg.cypher'}")
    print(f"[+] {out/'kg_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
