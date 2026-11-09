# SchemaSigma (SIGMA-KG)

Code for the paper *SchemaSigma: Schema-Constrained Generation of Sigma Detection Rules from Cyber Threat Intelligence Reports* (IEEE ISAICS 2026).

SchemaSigma turns a cyber threat intelligence (CTI) report into a Sigma detection rule. A knowledge graph mined from the public SigmaHQ rule set tells the language model which log sources, fields and modifiers are legal, and retrieved expert rules supply examples. A validator removes any field the graph does not allow.

## Contents

| File | What it does |
|---|---|
| `sk2_benchmark.py` | Builds the benchmark: pairs SigmaHQ rules with the CTI articles they cite and scores how answerable each article is |
| `sigma_kg_miner.py` | Mines the schema knowledge graph from SigmaHQ rules, excluding the benchmark rules |
| `sk2_kg.py` | Loads and queries the graph (JSON by default, Neo4j optional) |
| `sk2_split.py` | Article-grouped train/test split (seed 20261026) |
| `sk2_retrieval.py` | Exemplar retrieval (TF-IDF over training articles) |
| `sk2_extractor.py` | Schema-grounded generation and validation with an LLM |
| `sk2_evaluator.py` | All metrics, bootstrap CIs and paired tests |
| `sk2_run.py` | Runs the four configurations (`full`, `no_kg`, `no_retrieval`, `no_both`) |
| `baseline_nn.py` | Nearest-neighbour and majority baselines (no LLM) |
| `analysis/analyse_results.py` | Table II and all paired tests, from saved outputs |
| `analysis/paper_tables.py` | Table I and Table III, from saved outputs |
| `config/primary_exclusion_ids.json` | The 94 test ids removed to form the primary set (n = 132) |

## Setup

Python 3.10 or newer, and `git`.

```bash
git clone <this repository>
cd <this repository>
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set your API key in the environment (copy `.env.example` to `.env` if you keep keys there, but the scripts only read environment variables):

```bash
export OPENAI_API_KEY=sk-...         # Windows: set OPENAI_API_KEY=sk-...
```

## Reproducing the experiments

Run all commands from the repository root.

### 1. Get SigmaHQ at the commit used in the paper

The benchmark script clones the latest SigmaHQ if no copy exists. To use the same rules as the paper, clone it yourself first:

```bash
mkdir -p data_bench/_src
git clone https://github.com/SigmaHQ/sigma.git data_bench/_src/sigma
git -C data_bench/_src/sigma checkout da9bb07d642a2826e89702445d32c795209ec108
```

### 2. Build the benchmark (no LLM, no cost)

```bash
python sk2_benchmark.py --out data_bench --workers 6 --target 1000
```

This downloads the CTI articles cited by the rules. In the paper run it examined 2,782 of 3,757 rules and stopped at 1,000 usable report–rule pairs.

### 3. Mine the knowledge graph (no LLM, no cost)

```bash
python sigma_kg_miner.py --rules-dir data_bench/_src/sigma --exclude-index data_bench/index.json --out kg_out_clean
```

Expected at the pinned commit: 4,265 YAML files read, 1,468 benchmark rule files excluded, 2,582 rules mined, 185 log sources, 381 ATT&CK techniques.

### 4. Check everything before spending money

```bash
python sk2_run.py --bench data_bench --kg kg_out_clean/sigma_kg.json --configs full,no_kg,no_retrieval,no_both --strata grounded,partial --scope 7 --k 5 --model gpt-4o-mini --out results_dryrun --dry-run
```

`--dry-run` builds the split and runs all pre-flight checks, then stops before any API call.

### 5. Main experiment (GPT-4o-mini, paid)

```bash
python sk2_run.py --bench data_bench --kg kg_out_clean/sigma_kg.json --configs full,no_kg,no_retrieval,no_both --strata grounded,partial --scope 7 --k 5 --model gpt-4o-mini --compile-check --out results_stage1_main
```

### 6. Baselines (no LLM, no cost)

```bash
python baseline_nn.py --bench data_bench --kg kg_out_clean/sigma_kg.json --scope 7 --out results_stage1_baseline
```

### 7. Cross-model runs (paid, first 100 test reports)

GPT-4o:

```bash
python sk2_run.py --bench data_bench --kg kg_out_clean/sigma_kg.json --configs full,no_kg --strata grounded,partial --scope 7 --k 5 --model gpt-4o --test-limit 100 --compile-check --out results_stage1_gpt4o
```

Gemini 3.8 Flash, through Google's OpenAI-compatible endpoint. The script reads the key from `OPENAI_API_KEY`, so set it to your Gemini key for this run only:

```bash
OPENAI_API_KEY=$GEMINI_API_KEY python sk2_run.py --bench data_bench --kg kg_out_clean/sigma_kg.json --configs full,no_kg --strata grounded,partial --scope 7 --k 5 --model gemini-3.8-flash --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --test-limit 100 --compile-check --out results_stage1_gemini
```

### 8. Make the tables (no LLM, no cost)

Put the result folders from steps 5–7 in one directory (the repository root works), then:

```bash
python analysis/analyse_results.py . tables_out    # Table II and paired tests
python analysis/paper_tables.py . tables_out       # Tables I and III
```

## Expected results

Primary set, GPT-4o-mini, n = 132 (Tables I and II):

| | full | no_kg | no_retrieval | no_both | nn | majority |
|---|---|---|---|---|---|---|
| Field validity | 1.000 | 0.795 | 1.000 | 0.517 | 0.993 | 1.000 |
| Rules with an invalid field | 0 | 38 | 0 | 65 | | |
| Deployable rules | 92 | 67 | 78 | 20 | | |
| Log source exact | 0.697 | 0.644 | 0.667 | 0.394 | 0.432 | 0.629 |
| Field names F1 | 0.478 | 0.425 | 0.390 | 0.138 | 0.357 | 0.552 |
| Field+modifier F1 | 0.264 | 0.250 | 0.033 | 0.009 | 0.273 | 0.005 |
| Strict value F1 | 0.088 | 0.095 | 0.047 | 0.083 | 0.018 | 0 |
| Semantic value F1 | 0.220 | 0.194 | 0.122 | 0.195 | 0.034 | 0 |

Cross-model, `full` vs `no_kg` on the same 100 reports (Table III):

| | GPT-4o-mini | GPT-4o | Gemini 3.8 Flash |
|---|---|---|---|
| Field validity | 1.000 vs 0.797 | 1.000 vs 0.897 | 1.000 vs 0.985 |
| Rules with an invalid field | 0 vs 27 | 0 vs 13 | 0 vs 2 |
| Deployable rules | 73 vs 51 | 67 vs 62 | 75 vs 70 |

The two analysis scripts reproduce every number above exactly from the saved per-instance outputs of the paper run.

## How close a fresh run will be

Steps 2 and 5–7 depend on things outside this code, so a fresh run will be close to the paper but not identical:

- **Web articles change.** Step 2 downloads live pages. Some have moved, changed or gone offline since the paper run, which changes which pairs enter the benchmark.
- **LLM outputs are not fully deterministic**, even at temperature 0, and providers retire or update model versions.
- **SigmaHQ changes.** Pin the commit in step 1.

The graph-validity result does not depend on these: with the graph, every emitted field is legal by construction, whatever the model or article.

## Licences

SigmaHQ rules are under the Detection Rule License (DRL) 1.1. The CTI articles belong to their publishers and are not redistributed here; step 2 downloads them from their original URLs.

## Citation

If you use this code, please cite the paper (citation details will be added after publication).
