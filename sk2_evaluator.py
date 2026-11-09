"""
sk2_evaluator.py  --  TASK 5, part 1

Scoring, with the discipline the last two papers lacked.

Fixes carried over from src/sigma_evaluator.py:
  * recall gets its OWN numerator (the old code reused the precision count,
    which is why precision 0.4482 and recall 0.4483 came out near-identical)
  * fields, field+modifier and values are scored SEPARATELY, not pooled
  * substring matching is gone from the headline; it survives only as a
    clearly-labelled lenient variant
  * degenerate cases (either side empty) are excluded and counted, not scored 1.0

New here:
  * FIELD VALIDITY   - is the emitted field legitimate for the emitted log
                       source? A KG guarantee, reported as one.
  * LOGSOURCE ROUTING- exact and partial. This task does not exist in
                       LLMCloudHunter (all their rules are aws/cloudtrail).
  * SEMANTIC MATCH   - equivalence classes derived from the KG, published with
                       the paper, computed independently of any test instance.
                       Reported ALONGSIDE the strict number, never instead of it.
  * ENTITY EXTRACTION- LLMCloudHunter's own protocol, so the comparison is
                       apples to apples.

The rule on semantic matching, stated so a reviewer can check it:
    an equivalence class may only be built from the global SigmaHQ corpus and
    the value's own surface form. It may never consult the reference rule of
    the instance being scored.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

BLOCK_SKIP = {"condition", "timeframe"}

# Abstention markers written by sk2_run.assemble() and baseline_nn.py when the
# pipeline produced no usable selection. They are NOT field names and NOT
# values. Stage 1 fix (26 Oct 2026): field_validity used to count
# "__no_detection__" as an invented field, mixing abstention into the
# schema-hallucination number. Abstention is now its own metric.
PLACEHOLDER_FIELDS = {"__no_detection__"}
PLACEHOLDER_VALUES = {"__empty__"}


# ---------------------------------------------------------------- shared

def norm_value(v) -> str:
    s = str(v).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip("*").strip()


BINARY_EXT = (".exe", ".dll", ".ps1", ".bat", ".cmd", ".vbs", ".js", ".scr",
              ".com", ".sys", ".msi", ".jar", ".py", ".sh")


def canon_value(v) -> str:
    """
    Canonical form used ONLY by the semantic variant.
      C:\\Windows\\System32\\whoami.exe -> whoami
      \\whoami.exe                      -> whoami
      whoami                            -> whoami

    Surface-form normalisation only: path stripping and executable-extension
    stripping. It never consults the reference rule of the instance being
    scored, so it cannot leak.
    """
    s = norm_value(v).replace("\\", "/")
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    s = s.strip()
    for ext in BINARY_EXT:
        if s.endswith(ext):
            s = s[: -len(ext)]
            break
    return s.strip()


# Sigma filter blocks encode environment-specific ALLOWLISTS - benign paths and
# accounts a SOC excludes to cut false positives. They describe what is normal in
# one organisation, and are by definition absent from a public threat report.
# Measured across 3,144 SigmaHQ rules: 21.4% contain a filter block, and 12.3% of
# all detection values live inside one. Demanding them of a CTI-to-rule system
# measures something the source text cannot supply, so they are excluded by
# default and the choice is reported.
EXCLUDE_FILTER_BLOCKS = True


def _is_filter_block(name) -> bool:
    n = str(name).strip().lower()
    return n.startswith("filter") or n.startswith("exclusion") or n.endswith("_filter")


def split_detection(det, exclude_filters: bool | None = None):
    """-> (fields, field|modifier, values) for one detection block set."""
    if exclude_filters is None:
        exclude_filters = EXCLUDE_FILTER_BLOCKS
    fields, fmods, values = set(), set(), set()

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                ks = str(k).strip()
                if ks.lower() in BLOCK_SKIP:
                    continue
                base = ks.split("|")[0].strip().lower()
                if base:
                    fields.add(base)
                    fmods.add(ks.strip().lower())
                walk(v)
        elif isinstance(node, list):
            for i in node:
                walk(i)
        elif isinstance(node, (str, int, float, bool)):
            s = norm_value(node)
            if len(s) > 1:
                values.add(s)

    if isinstance(det, dict):
        for bname, block in det.items():
            if str(bname).lower() in BLOCK_SKIP:
                continue
            if exclude_filters and _is_filter_block(bname):
                continue
            walk(block)
    return fields, fmods, values


def split_tuples(det, exclude_filters: bool | None = None):
    """
    -> (field_value, fieldmod_value) sets of tuples, Stage 1 fix (26 Oct 2026).

    split_detection() returns fields and values as SEPARATE sets, so a value
    counted as matched even when it sat under a different field than in the
    reference (Image: -enc / CommandLine: powershell.exe scored like the
    correct rule). These tuples keep the association:
        field_value     ("image", "\\adfind.exe")
        fieldmod_value  ("image|endswith", "\\adfind.exe")
    Same normalisation, filter-block exclusion and BLOCK_SKIP as
    split_detection, and abstention placeholders are dropped. Keyless list
    items (Sigma keyword search) are stored under the pseudo-field "|keywords".
    """
    if exclude_filters is None:
        exclude_filters = EXCLUDE_FILTER_BLOCKS
    fv, fmv = set(), set()

    def walk(node, key):
        if isinstance(node, dict):
            for k, v in node.items():
                ks = str(k).strip()
                if ks.lower() in BLOCK_SKIP:
                    continue
                walk(v, ks.lower())
        elif isinstance(node, list):
            for i in node:
                walk(i, key)
        elif isinstance(node, (str, int, float, bool)):
            s = norm_value(node)
            if len(s) <= 1 or s in PLACEHOLDER_VALUES:
                return
            k = key or "|keywords"
            base = k.split("|")[0].strip()
            if base in PLACEHOLDER_FIELDS:
                return
            fv.add((base, s))
            fmv.add((k, s))

    if isinstance(det, dict):
        for bname, block in det.items():
            if str(bname).lower() in BLOCK_SKIP:
                continue
            if exclude_filters and _is_filter_block(bname):
                continue
            walk(block, None)
    return fv, fmv


def is_abstention(det) -> bool:
    """True when the rule carries no real field (only the abstention marker)."""
    fields, _, _ = split_detection(det or {})
    return bool(fields) and not (fields - PLACEHOLDER_FIELDS)


def prf(pred: set, gt: set):
    """Precision and recall with INDEPENDENT numerators."""
    if not pred or not gt:
        return 0.0, 0.0, 0.0, True
    p = len(pred & gt) / len(pred)
    r = len(gt & pred) / len(gt)
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f, False


def prf_semantic(pred: set, gt: set):
    """Same, over canonicalised values. Symmetric, so recall stays independent."""
    if not pred or not gt:
        return 0.0, 0.0, 0.0, True
    P = {canon_value(x) for x in pred}
    G = {canon_value(x) for x in gt}
    p = len(P & G) / len(P)
    r = len(G & P) / len(G)
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f, False


# ---------------------------------------------------------------- metrics

def logsource_scores(pred_ls: dict, gt_ls: dict):
    keys = [k for k in ("product", "category", "service") if (gt_ls or {}).get(k)]
    if not keys:
        return {"ls_exact": None, "ls_partial": None, "ls_keys": 0}
    hits = 0
    for k in keys:
        a, b = (pred_ls or {}).get(k), (gt_ls or {}).get(k)
        if a and str(a).strip().lower() == str(b).strip().lower():
            hits += 1
    return {"ls_exact": 1.0 if hits == len(keys) else 0.0,
            "ls_partial": hits / len(keys), "ls_keys": len(keys)}


def field_validity(pred_rule: dict, kg) -> dict:
    """
    Fraction of emitted fields that are legitimate for the log source the rule
    itself declares. This is the anti-hallucination number LLMCloudHunter never
    reports. For our KG-constrained pipeline it is 1.0 by construction; for any
    baseline it is a real measurement.
    """
    ls = pred_rule.get("logsource") or {}
    p, c, s = ls.get("product"), ls.get("category"), ls.get("service")
    fields, _, _ = split_detection(pred_rule.get("detection") or {})
    # abstention marker is not a field: drop it before checking (Stage 1 fix)
    fields = fields - PLACEHOLDER_FIELDS
    if not fields:
        return {"field_validity": None, "n_fields": 0, "invalid_fields": []}
    allowed = {a.lower() for a in kg.allowed_fields(p, c, s)}
    if not allowed:
        return {"field_validity": None, "n_fields": len(fields), "invalid_fields": []}
    bad = sorted(f for f in fields if f not in allowed)
    return {"field_validity": 1.0 - len(bad) / len(fields),
            "n_fields": len(fields), "invalid_fields": bad}


def evaluate(pred_rule: dict, gt_rule: dict, kg=None) -> dict:
    pf, pfm, pv = split_detection(pred_rule.get("detection") or {})
    gf, gfm, gv = split_detection(gt_rule.get("detection") or {})

    f_p, f_r, f_f1, f_deg = prf(pf, gf)
    m_p, m_r, m_f1, m_deg = prf(pfm, gfm)
    v_p, v_r, v_f1, v_deg = prf(pv, gv)
    sv_p, sv_r, sv_f1, sv_deg = prf_semantic(pv, gv)
    pfv, pfmv = split_tuples(pred_rule.get("detection") or {})
    gfv, gfmv = split_tuples(gt_rule.get("detection") or {})
    # Degenerate ONLY when the reference has no tuple. An empty prediction
    # (abstention) against a non-empty reference scores 0 and is kept, so a
    # configuration cannot raise its mean by abstaining on hard instances.
    fv_p, fv_r, fv_f1, _ = prf(pfv, gfv)
    fmv_p, fmv_r, fmv_f1, _ = prf(pfmv, gfmv)
    fv_deg = fmv_deg = not gfv
    abstained = is_abstention(pred_rule.get("detection"))
    n_real = len(pf - PLACEHOLDER_FIELDS)

    pt = {str(t).lower().replace("attack.", "") for t in (pred_rule.get("tags") or [])}
    gt_t = {str(t).lower().replace("attack.", "") for t in (gt_rule.get("tags") or [])}
    tid = re.compile(r"^t\d{4}(\.\d{3})?$")
    _, _, tech_f1, tech_deg = prf({t for t in pt if tid.match(t)},
                                  {t for t in gt_t if tid.match(t)})

    out = {
        "field_p": f_p, "field_r": f_r, "field_f1": f_f1, "field_deg": f_deg,
        "fieldmod_f1": m_f1, "fieldmod_deg": m_deg,
        "value_p": v_p, "value_r": v_r, "value_f1": v_f1, "value_deg": v_deg,
        "value_sem_p": sv_p, "value_sem_r": sv_r, "value_sem_f1": sv_f1, "value_sem_deg": sv_deg,
        "tech_f1": tech_f1, "tech_deg": tech_deg,
        # Stage 1 (26 Oct 2026): value must sit on the same field (and modifier)
        "fieldvalue_p": fv_p, "fieldvalue_r": fv_r, "fieldvalue_f1": fv_f1,
        "fieldvalue_deg": fv_deg,
        "fieldmodvalue_p": fmv_p, "fieldmodvalue_r": fmv_r,
        "fieldmodvalue_f1": fmv_f1, "fieldmodvalue_deg": fmv_deg,
        "abstained": 1.0 if abstained else 0.0,
        "n_real_pred_fields": n_real,
        "n_pred_fields": len(pf), "n_gt_fields": len(gf),
        "n_pred_values": len(pv), "n_gt_values": len(gv),
    }
    out.update(logsource_scores(pred_rule.get("logsource") or {}, gt_rule.get("logsource") or {}))
    if kg is not None:
        out.update(field_validity(pred_rule, kg))
    return out


# --------------------------------------------- LLMCloudHunter's protocol

def _wild(pattern: str, candidate: str) -> bool:
    rx = "^" + re.escape(pattern).replace(r"\*", ".*") + "$"
    return re.match(rx, candidate) is not None


def entity_prf(gt_items: set, pred_items: set):
    """
    Set-level, wildcard-tolerant comparison -- LLMCloudHunter's _compare_data.
    Reimplemented here so our numbers sit in the same table as theirs.
    """
    if not gt_items and not pred_items:
        return 1.0, 1.0, 1.0
    tp, matched = 0, set()
    for g in gt_items:
        for p in pred_items:
            if _wild(g, p):
                tp += 1
                matched.add(p)
                break
    fp = len(pred_items - matched)
    fn = len(gt_items) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


# ---------------------------------------------------------------- stats

def bootstrap_ci(vals, n_boot: int = 10000, seed: int = 42):
    v = [float(x) for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return 0.0, 0.0, 0.0, 0
    a = np.asarray(v)
    if len(a) == 1:
        return round(float(a[0]), 4), round(float(a[0]), 4), round(float(a[0]), 4), 1
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return (round(float(a.mean()), 4), round(float(np.percentile(m, 2.5)), 4),
            round(float(np.percentile(m, 97.5)), 4), len(a))


def paired_test(a_vals, b_vals):
    """Wilcoxon on REAL paired per-instance scores. Never on reconstructed data."""
    from scipy import stats as st
    a = np.asarray([float(x) for x in a_vals], dtype=float)
    b = np.asarray([float(x) for x in b_vals], dtype=float)
    d = a - b
    if not len(d) or not np.any(d != 0):
        return {"mean_diff": 0.0, "p": 1.0, "wins": 0, "losses": 0,
                "ties": int(len(d)), "n": int(len(d))}
    try:
        p = float(st.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception:
        p = float("nan")
    return {"mean_diff": float(d.mean()), "p": p,
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
            "ties": int((d == 0).sum()), "n": int(len(d))}


def aggregate(rows, metric, deg_key=None):
    vals = [r.get(metric) for r in rows
            if not (deg_key and r.get(deg_key)) and r.get(metric) is not None]
    return bootstrap_ci(vals)


# =====================================================================
#  METRIC SUITE v2  --  built to answer the four reviews directly
# =====================================================================
#
# AxK1 #2  "distinguish correctness ensured by design from performance
#           achieved through model reasoning"
#        -> GUARANTEE metrics vs CAPABILITY metrics, reported apart.
#
# eVJJ #5  "'No KG Prior' has the same Key F1 and Value Ratio but much
#           worse product/service match ... the discussion does not
#           analyze this nuance"
#        -> CONDITIONAL decomposition. Field F1 is reported again
#           restricted to correctly-routed instances, which separates the
#           graph's routing effect from its vocabulary effect.
#
# FMrU #5  "fails to explain the real boundary between neural extraction
#           errors and symbolic structural errors"
#        -> ERROR ATTRIBUTION. Every failed instance is assigned to one
#           of six causes, three neural and three symbolic.
#
# eVJJ #6  "rule deployability ... not evaluated"
#        -> DEPLOYABLE RATE, a composite an operator would actually care
#           about.
#
# Nothing here consults the reference rule during generation. All of it
# runs at scoring time only.


# ---------------------------------------------------------------- 1. routing
# Log-source accuracy alone is misleading: windows/process_creation is
# 37.3% of SigmaHQ, so a system that always guesses it scores 0.37 while
# knowing nothing. Macro-F1 averages over classes instead of instances and
# cannot be gamed that way. Report both, plus the majority baseline.

def _norm_ls(s):
    """
    Normalise a 'product/category/service' string for comparison.

    This existed as a bug: the comparison was case-sensitive, so a model that
    answered 'Windows/process_creation/-' scored zero against the reference
    'windows/process_creation/-'. The unconstrained configuration capitalises
    product names, which made its routing accuracy read 0.0231 instead of
    0.5833. Everything downstream of that number was wrong.
    """
    return "/".join(part.strip().lower() for part in str(s or "").split("/"))


def macro_routing_f1(rows, pred_key="pred_logsource", gt_key="gt_logsource"):
    """Macro-averaged P/R/F1 over log-source classes, plus the majority baseline."""
    from collections import Counter
    pairs = [(_norm_ls(r.get(pred_key)), _norm_ls(r.get(gt_key))) for r in rows
             if r.get(gt_key) not in (None, "", "None")]
    if not pairs:
        return {"macro_p": 0.0, "macro_r": 0.0, "macro_f1": 0.0,
                "accuracy": 0.0, "majority_baseline": 0.0, "n_classes": 0, "n": 0}

    classes = sorted({g for _, g in pairs})
    tp = Counter(); fp = Counter(); fn = Counter()
    for p, g in pairs:
        if p == g:
            tp[g] += 1
        else:
            fn[g] += 1
            if p:
                fp[p] += 1

    ps, rs, fs = [], [], []
    for c in classes:
        p_ = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r_ = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f_ = (2 * p_ * r_ / (p_ + r_)) if (p_ + r_) else 0.0
        ps.append(p_); rs.append(r_); fs.append(f_)

    gt_counts = Counter(g for _, g in pairs)
    majority = max(gt_counts.values()) / len(pairs)
    return {"macro_p": round(sum(ps) / len(classes), 4),
            "macro_r": round(sum(rs) / len(classes), 4),
            "macro_f1": round(sum(fs) / len(classes), 4),
            "accuracy": round(sum(1 for p, g in pairs if p == g) / len(pairs), 4),
            "majority_baseline": round(majority, 4),
            "n_classes": len(classes), "n": len(pairs)}


# ---------------------------------------------------------------- 2. conditional
# The decomposition a reviewer can follow:
#     P(good rule) = P(right logsource)
#                  x P(right fields | right logsource)
#                  x P(right values | right fields)
# Each stage is measured separately, so a gain can be attributed to the
# stage that produced it instead of being smeared across the total.

def conditional_metrics(rows):
    routed = [r for r in rows if r.get("ls_exact") == 1.0]
    out = {"n_routed_correct": len(routed), "n": len(rows)}
    for m, deg in (("field_f1", "field_deg"), ("fieldmod_f1", "fieldmod_deg"),
                   ("value_sem_f1", "value_sem_deg"), ("value_f1", "value_deg")):
        vals = [r[m] for r in routed
                if r.get(m) is not None and not r.get(deg)]
        out[f"{m}_given_routed"] = round(sum(vals) / len(vals), 4) if vals else None
    fielded = [r for r in routed if (r.get("field_f1") or 0) >= 0.999]
    vals = [r["value_sem_f1"] for r in fielded
            if r.get("value_sem_f1") is not None and not r.get("value_sem_deg")]
    out["n_fields_correct"] = len(fielded)
    out["value_sem_f1_given_fields"] = round(sum(vals) / len(vals), 4) if vals else None
    return out


# ---------------------------------------------------------------- 3. hallucination
# Three distinct kinds. Previous work reports none of them separately,
# because detecting the first two requires a schema graph to compare against.

def hallucination_rates(rows):
    def mean(key, default=None):
        v = [float(r[key]) for r in rows if r.get(key) is not None]
        return round(sum(v) / len(v), 4) if v else default

    n = max(1, len(rows))
    invalid = [1.0 - float(r["field_validity"]) for r in rows
               if r.get("field_validity") is not None]
    tot_prop = sum(int(r.get("n_observables") or 0) +
                   int(r.get("n_ungrounded_values") or 0) for r in rows)
    tot_ungr = sum(int(r.get("n_ungrounded_values") or 0) for r in rows)
    return {
        # a field name that does not exist for the declared log source
        "schema_hallucination_rate": round(sum(invalid) / len(invalid), 4) if invalid else None,
        # a log source that does not exist in SigmaHQ at all
        "logsource_hallucination_rate": mean("logsource_snapped"),
        # a value the model could not point to anywhere in the report
        "value_hallucination_rate": round(tot_ungr / tot_prop, 4) if tot_prop else 0.0,
        # Stage 1: abstention reported on its own, never inside the rate above
        "abstention_rate": mean("abstained"),
        # schema padding: fields emitted with no evidence behind them
        "empty_field_rate": round(
            sum(int(r.get("n_empty_fields") or 0) for r in rows) / n, 4),
    }


# ---------------------------------------------------------------- 4. deployable
# What an operator cares about: can this rule be put into a SIEM at all?
# Not "is it perfect" - "is it loadable and pointed at the right telemetry".

def deployable_rate(rows, require_compile=True):
    ok = 0
    counted = 0
    for r in rows:
        if require_compile and r.get("compiles_splunk") is None:
            continue
        counted += 1
        compiles = (r.get("compiles_splunk") is None) or (r.get("compiles_splunk") == 1.0)
        valid = (r.get("field_validity") is None) or (float(r["field_validity"]) >= 0.999)
        routed = r.get("ls_exact") == 1.0
        # Stage 1: the abstention marker is not a field. Fall back to the old
        # column only for rows scored before n_real_pred_fields existed.
        nreal = r.get("n_real_pred_fields")
        nonempty = (int(float(nreal)) if nreal not in (None, "") else
                    int(float(r.get("n_pred_fields") or 0))) > 0 \
            and not float(r.get("abstained") or 0)
        if compiles and valid and routed and nonempty:
            ok += 1
    return {"deployable_rate": round(ok / counted, 4) if counted else None,
            "n_scored": counted}


# ---------------------------------------------------------------- 5. error attribution
# FMrU asked for "the real boundary between neural extraction errors and
# symbolic structural errors". Six causes, three of each. Every instance
# that is not fully correct receives exactly one label, taking the FIRST
# that applies, so the categories partition the failures.

ERROR_CLASSES = [
    ("ok",                     "rule matched the reference on log source and fields"),
    ("neural_hallucinated",    "model proposed values not present in the report"),
    ("neural_missed",          "the values were in the report and the model did not extract them"),
    ("neural_misrouted",       "model chose a real but wrong log source"),
    ("symbolic_misrouted",     "model's log source did not exist; the graph snapped to a wrong one"),
    ("symbolic_dropped",       "the graph removed a field that the reference rule actually uses"),
    ("out_of_scope",           "the reference uses a log source or field the graph does not cover"),
]


def attribute_error(row, pred_rule=None, gt_rule=None, kg=None, article_text=None):
    """
    Assign one cause to one instance. Uses only the row's own record plus the
    reference rule - i.e. scoring-time information, never generation-time.
    """
    routed_ok = row.get("ls_exact") == 1.0
    field_f1 = row.get("field_f1")
    fields_ok = field_f1 is not None and field_f1 >= 0.999

    if routed_ok and fields_ok:
        return "ok"

    # --- routing failures: whose fault? -----------------------------
    if not routed_ok:
        # the graph had to correct the model, and the correction was wrong
        if row.get("logsource_snapped") == 1.0:
            return "symbolic_misrouted"
        # is the reference log source even representable?
        if kg is not None and gt_rule is not None:
            gls = (gt_rule.get("logsource") or {})
            if not kg.allowed_fields(gls.get("product"), gls.get("category"),
                                     gls.get("service")):
                return "out_of_scope"
        return "neural_misrouted"

    # --- routing was right, so the failure is in the fields ---------
    dropped = row.get("dropped_fields") or []
    if isinstance(dropped, str):
        # Field names contain '|' themselves ("CommandLine|contains"), so the
        # list must not be joined with '|'. It is written with ';;'.
        sep = ";;" if ";;" in dropped else None
        dropped = [d for d in (dropped.split(sep) if sep else [dropped]) if d]
    if gt_rule is not None and dropped:
        gt_fields, _, _ = split_detection(gt_rule.get("detection") or {})
        if any(str(d).split("|")[0].strip().lower() in gt_fields for d in dropped):
            # the graph threw away a field the reference actually needs
            return "symbolic_dropped"

    if int(row.get("n_ungrounded_values") or 0) > 0:
        return "neural_hallucinated"

    return "neural_missed"


def error_profile(rows_with_context):
    """rows_with_context: list of (row, gt_rule) pairs, plus an optional kg."""
    from collections import Counter
    c = Counter()
    for item in rows_with_context:
        row, gt_rule, kg = (item + (None,) * 3)[:3]
        c[attribute_error(row, gt_rule=gt_rule, kg=kg)] += 1
    total = max(1, sum(c.values()))
    neural = sum(v for k, v in c.items() if k.startswith("neural_"))
    symbolic = sum(v for k, v in c.items() if k.startswith("symbolic_"))
    return {"counts": dict(c),
            "share": {k: round(v / total, 4) for k, v in c.items()},
            "neural_share": round(neural / total, 4),
            "symbolic_share": round(symbolic / total, 4),
            "n": total}
