"""Analysis behind Table II and the paired tests in the paper (27 Oct 2026): primary = in-scope test pairs minus the 94 frozen
previously-evaluated ids; sensitivity = all 194. Reads per_instance.csv files only."""
import csv, json, sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sk2_evaluator import aggregate, paired_test
# usage: python analysis/analyse_results.py <results_dir> <output_dir>
# <results_dir> must contain results_stage1_main/ and results_stage1_baseline/
U = sys.argv[1]; OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rows = list(csv.DictReader(open(f'{U}/results_stage1_main/per_instance.csv', encoding='utf-8')))
rows += list(csv.DictReader(open(f'{U}/results_stage1_baseline/per_instance.csv', encoding='utf-8')))
ex = set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'primary_exclusion_ids.json')))['ids'])
def num(v):
    if v in (None, '', 'None', 'nan'): return None
    if v in ('True', 'False'): return 1.0 if v == 'True' else 0.0
    try: return float(v)
    except ValueError: return v
rows = [{k: num(v) if k not in ('config', 'id', 'stratum') else v for k, v in r.items()} for r in rows]
M = [('ls_exact', None), ('field_validity', None), ('field_f1', 'field_deg'), ('fieldmod_f1', 'fieldmod_deg'),
     ('value_f1', 'value_deg'), ('value_sem_f1', 'value_sem_deg'), ('fieldvalue_f1', 'fieldvalue_deg'),
     ('fieldmodvalue_f1', 'fieldmodvalue_deg'), ('tech_f1', 'tech_deg'), ('abstained', None)]
cfgs = ['full', 'no_kg', 'no_retrieval', 'no_both', 'nn', 'majority']
failed = sorted({r['id'] for r in rows if r.get('generation_failed') in (1.0, True)})
res = {'excluded_ids': len(ex), 'failed_generation_ids': failed, 'subsets': {}}
L = ['# Stage 1 results (article-grouped split, clean graph, gpt-4o-mini, T=0)', '',
     f'Failed generations (API error), scored as produced: {failed}', '']
for sub in ('primary', 'sensitivity_all'):
    keep = lambda r: sub == 'sensitivity_all' or r['id'] not in ex
    R = [r for r in rows if keep(r)]
    t = res['subsets'][sub] = {'means': {}, 'paired': {}}
    n = len({r['id'] for r in R if r['config'] == 'full'})
    L += [f'## {sub} (n = {n})', '', '| metric | ' + ' | '.join(cfgs) + ' |', '|---|' + '---|' * len(cfgs)]
    for m, d in M:
        cells = []
        for c in cfgs:
            mean, lo, hi, k = aggregate([r for r in R if r['config'] == c], m, d)
            t['means'].setdefault(c, {})[m] = {'mean': mean, 'ci_low': lo, 'ci_high': hi, 'n': k}
            cells.append(f'{mean:.3f} [{lo:.3f}, {hi:.3f}]' if k else '-')
        L.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    L += ['', 'Paired Wilcoxon, full minus other (instances where both are scored):', '',
          '| comparison | metric | mean diff | p | W/L/T | n |', '|---|---|---|---|---|---|']
    byc = {c: {r['id']: r for r in R if r['config'] == c} for c in cfgs}
    for o in ['no_kg', 'no_retrieval', 'no_both', 'nn', 'majority']:
        for m, d in M:
            ids = [i for i in byc['full'] if i in byc[o] and byc['full'][i].get(m) is not None
                   and byc[o][i].get(m) is not None and not (d and (byc['full'][i].get(d) or byc[o][i].get(d)))]
            if not ids: continue
            p = paired_test([byc['full'][i][m] for i in ids], [byc[o][i][m] for i in ids])
            t['paired'][f'full_vs_{o}|{m}'] = p
            L.append(f"| full vs {o} | {m} | {p['mean_diff']:+.3f} | {p['p']:.2g} | {p['wins']}/{p['losses']}/{p['ties']} | {p['n']} |")
    L.append('')
json.dump(res, open(f'{OUT}/stage1_analysis.json', 'w'), indent=1)
open(f'{OUT}/STAGE1_RESULTS_TABLES.md', 'w').write('\n'.join(L))
print('\n'.join(L))
