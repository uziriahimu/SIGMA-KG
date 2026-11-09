"""Structural table (Table I) and cross-model table (Table III) of the paper.

Reads saved per_instance.csv files only; makes no API call.

usage: python analysis/paper_tables.py <results_dir> <output_dir>
<results_dir> must contain results_stage1_main/; results_stage1_gpt4o/ and
results_stage1_gemini/ are used for Table III when present.
"""
import ast, csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from sk2_evaluator import deployable_rate

U, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
EX = set(json.load(open(os.path.join(HERE, '..', 'config', 'primary_exclusion_ids.json')))['ids'])


def num(v):
    if v in (None, '', 'None', 'nan'): return None
    if v in ('True', 'False'): return 1.0 if v == 'True' else 0.0
    try: return float(v)
    except ValueError: return v


def load(name):
    p = os.path.join(U, name, 'per_instance.csv')
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    return [{k: (v if k in ('config', 'id', 'stratum', 'invalid_fields') else num(v))
             for k, v in r.items()} for r in rows]


def structural(rows):
    """Field validity, pooled invalid fields, rules with an invalid field,
    abstention, Splunk compile rate and deployable count for one config."""
    n = len(rows)
    fv = [r['field_validity'] for r in rows if r['field_validity'] is not None]
    inv = tot = bad_rules = 0
    for r in rows:
        bad = ast.literal_eval(r['invalid_fields']) if r['invalid_fields'] else []
        bad_rules += bool(bad)
        if r['field_validity'] is not None:      # log source exists in the graph
            inv += len(bad); tot += int(r['n_real_pred_fields'] or 0)
    dep = deployable_rate(rows)
    return {'n': n,
            'field_validity': round(sum(fv) / len(fv), 3) if fv else None,
            'invalid_fields': f'{inv}/{tot}', 'invalid_fields_pct': round(100 * inv / tot, 1) if tot else 0.0,
            'rules_with_invalid': bad_rules,
            'abstained': int(sum(r['abstained'] or 0 for r in rows)),
            'compiles_splunk_pct': round(100 * sum(r['compiles_splunk'] == 1.0 for r in rows) / n, 1),
            'deployable': round(dep['deployable_rate'] * dep['n_scored']),
            'strict_value_f1': round(sum(r['value_f1'] or 0 for r in rows) / n, 3),
            'ls_exact': round(sum(r['ls_exact'] or 0 for r in rows) / n, 2)}


res, L = {}, []
main = load('results_stage1_main')
prim = [r for r in main if r['id'] not in EX]
L += ['# Table I: structural quality, primary set (gpt-4o-mini)', '',
      '| config | n | field validity | invalid fields | rules w/ invalid | abstained | compiles (Splunk) % | deployable |',
      '|---|---|---|---|---|---|---|---|']
for c in ['full', 'no_kg', 'no_retrieval', 'no_both']:
    s = res.setdefault('table1', {})[c] = structural([r for r in prim if r['config'] == c])
    L.append(f"| {c} | {s['n']} | {s['field_validity']:.3f} | {s['invalid_fields']} ({s['invalid_fields_pct']}%) | "
             f"{s['rules_with_invalid']} | {s['abstained']} | {s['compiles_splunk_pct']} | {s['deployable']} |")

runs = [('gpt-4o-mini', main), ('gpt-4o', load('results_stage1_gpt4o')), ('gemini', load('results_stage1_gemini'))]
if runs[1][1]:
    # the cross-model runs score the first 100 test reports (--test-limit 100);
    # gpt-4o-mini is restricted to the same ids
    ids = {r['id'] for r in runs[1][1] if r['config'] == 'full'}
    L += ['', f'# Table III: full vs no_kg across LLMs (same {len(ids)} reports)', '',
          '| model | config | field validity | rules w/ invalid | strict value F1 | log source exact | deployable | abstained |',
          '|---|---|---|---|---|---|---|---|']
    for m, rows in runs:
        if not rows: continue
        for c in ['full', 'no_kg']:
            s = structural([r for r in rows if r['config'] == c and r['id'] in ids])
            res.setdefault('table3', {}).setdefault(m, {})[c] = s
            L.append(f"| {m} | {c} | {s['field_validity']:.3f} | {s['rules_with_invalid']} | {s['strict_value_f1']:.3f} | "
                     f"{s['ls_exact']:.2f} | {s['deployable']} | {s['abstained']} |")

json.dump(res, open(os.path.join(OUT, 'paper_tables.json'), 'w'), indent=1)
open(os.path.join(OUT, 'PAPER_TABLES_I_III.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
