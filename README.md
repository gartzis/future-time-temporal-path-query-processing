# Future-Time Shortest Temporal Path Query Processing under Prediction Uncertainty

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20644952.svg)](https://doi.org/10.5281/zenodo.20644952)

Research code for reproducing the experiments of the paper:

**Future-Time Shortest Temporal Path Query Processing under Prediction Uncertainty**

The paper studies future-time shortest temporal path queries. A query is issued at the current timestamp, but the answer concerns a future timestamp and may depend on edges that have not yet appeared. The code combines temporal query processing with prediction oracles, constructs candidate future temporal paths, and ranks them by estimated shortest-path probability.

## Repository overview

The repository contains:

* controlled-oracle experiments for candidate-space recovery and probability-based path selection
* real-oracle experiments with N2VLP-Static, TGN, and JODIE
* future-time temporal path construction and ranking
* overlap-aware shortest-path probability estimation
* cache-aware query processing
* scripts for reproducing the paper tables and figures

## Quick start

Install the requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run one experiment from the repository root:

```bash
python Experiments/run_RQ1_candidate_space_oracle.py
```

All scripts use editable global parameters at the top of the file. There are no command-line arguments.

For example, to run a smaller check before a full experiment, edit the corresponding script and set:

```python
MAX_TEST_ROWS_PER_DATASET = 5
```

or:

```python
MAX_TESTS_PER_DATASET = 5
```

depending on the script.

## Repository layout

```text
.
├── main.py
├── oracle_methods.py
├── pipeline_methods.py
├── Experiments/
├── Plotting/
├── FuturePathEstimator/
├── Helpers/
├── Data/
├── External/
└── Results/
```

## Main files

### `main.py`

Main runner for the real-oracle pipeline. Edit the global parameters at the top of the file and run:

```bash
python main.py
```

This file runs the selected dataset and selected real oracle.

### `oracle_methods.py`

Function-based implementations for the real prediction oracles:

* N2VLP-Static
* TGN-MaxTime
* TGN-PerTime
* JODIE-Frozen
* JODIE-Update

The oracle code is function-based and does not use oracle classes.

### `pipeline_methods.py`

Shared implementation methods used by `main.py` and `oracle_methods.py`. It contains the main training, scoring, candidate generation, path update, path ranking, timing, and output functions.

### `FuturePathEstimator/`

PredictSP and Luby-Karp based path-probability estimation code.

### `Helpers/`

Original helper code for temporal edge-stream processing.

### `External/`

Vendored external implementations used by the real-oracle experiments:

* TGN
* JODIE

## Experiments

The paper uses five research questions. Each one has a corresponding script.

```bash
python Experiments/run_RQ1_candidate_space_oracle.py
python Experiments/run_RQ2_optimal_temporal_edge_oracle.py
python Experiments/run_RQ3_oracle_quality.py
python Experiments/run_RQ4_full_pipeline.py
python Experiments/run_RQ5_runtime_cache.py
```

### RQ1: Candidate-space oracle

```bash
python Experiments/run_RQ1_candidate_space_oracle.py
```

This script runs the standalone candidate-space audit used for the landmark sensitivity experiment.

It does not train or call a prediction model. It checks whether the true future temporal path can be recovered from the generated candidate space when generated candidate edges are accepted with a fixed score.

The full paper setting uses:

```python
LANDMARK_POOL_SIZES_TO_RUN = [1, 3, 5]
MAX_TEST_ROWS_PER_DATASET = None
```

For a quick check, use:

```python
DATASETS = ["enron"]
LANDMARK_POOL_SIZES_TO_RUN = [5]
MAX_TEST_ROWS_PER_DATASET = 5
```

### RQ2: Optimal temporal-edge oracle

```bash
python Experiments/run_RQ2_optimal_temporal_edge_oracle.py
```

This script runs the standalone optimal temporal-edge oracle experiment.

It bypasses candidate generation and uses only true future temporal edges at their actual timestamps. It varies the controlled edge probability and the shortest-path threshold, then writes the prediction files used for the coverage and exact-match heatmaps.

The full paper setting uses:

```python
EDGE_PROBAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHORTEST_PROBA_BOUNDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_TESTS_PER_DATASET = None
```

For a quick check, use:

```python
DATASETS = ["enron"]
EDGE_PROBAS = [0.5]
SHORTEST_PROBA_BOUNDS = [0.0]
MAX_TESTS_PER_DATASET = 5
```

### RQ3: Real oracle quality

```bash
python Experiments/run_RQ3_oracle_quality.py
```

This script evaluates the prediction quality of the real oracles before final temporal path ranking.

It reports metrics such as:

* AUC for edge-scoring oracles
* MRR and Recall@10 for target-ranking oracles

### RQ4: Real-oracle full pipeline

```bash
python Experiments/run_RQ4_full_pipeline.py
```

This script runs the complete future-time query-processing pipeline using real prediction oracles.

It evaluates the ranked future temporal paths using top-k path metrics, including weighted path edit error, normalized distance error, normalized timestamp error, coverage, recall, and MRR.

### RQ5: Runtime and cache efficiency

```bash
python Experiments/run_RQ5_runtime_cache.py
```

This script measures training time, query-processing time, and cache behavior.

It compares cache-enabled and cache-disabled execution for the real-oracle pipeline.

## Real oracle names

The real-oracle pipeline supports the following oracle names:

| Code name      | Paper name   | Oracle type           | Temporal information                                                                                         |
| -------------- | ------------ | --------------------- | ------------------------------------------------------------------------------------------------------------ |
| `n2vlp_static` | N2VLP-Static | Edge-scoring oracle   | Non-temporal. Scores node pairs using static Node2Vec embeddings computed on the observed prefix graph.      |
| `tgn_max_time` | TGN-MaxTime  | Edge-scoring oracle   | Temporal. Scores candidate edges across future timestamps and keeps the timestamp with the maximum score.    |
| `tgn_per_time` | TGN-PerTime  | Edge-scoring oracle   | Temporal. Scores candidate edges separately at each rollout timestamp.                                       |
| `jodie_frozen` | JODIE-Frozen | Target-ranking oracle | Temporal model. Uses the observed-prefix JODIE state and keeps it fixed during the future rollout.           |
| `jodie_update` | JODIE-Update | Target-ranking oracle | Temporal model. Updates the query-local JODIE state with accepted predicted interactions during the rollout. |

To change the oracle in `main.py`, edit:

```python
ORACLE_NAME = "n2vlp_static"
```


## Input data

Temporal edge streams are expected under:

```text
Data/Datasets/
```

Expected files:

```text
Data/Datasets/enron.csv
Data/Datasets/email-eu.csv
Data/Datasets/collegemsg.csv
Data/Datasets/bitcoin.csv
```

Each file should contain source, destination, and timestamp columns. Common column names are accepted:

* source column: `source`, `src`, `u`
* destination column: `destination`, `target`, `dst`, `v`
* time column: `time`, `timestamp`, `ts`, `t`

Query-test files are expected under:

```text
Data/query_tests/
```

Expected files:

```text
Data/query_tests/enron.tsv
Data/query_tests/email_eu.tsv
Data/query_tests/collegemsg.tsv
Data/query_tests/bitcoin.tsv
```

Each query-test file must contain:

```text
source destination prev_Path future_Path time
```

The path columns store comma-separated node ids.

## Outputs

Experiment outputs are written under:

```text
Results/
```

Depending on the selected experiment, the code writes:

* per-query predictions
* candidate-recovery summaries
* probability-threshold summaries
* accepted predicted edges
* weighted top-k path metrics
* runtime breakdowns
* cache counters
* heatmap values
* LaTeX table files

## Plotting

Plotting scripts are stored under:

```text
Plotting/
```

They use the output files written under `Results/` to reproduce the figures and tables used in the paper.

## Reproducibility notes

The code follows a simple research-script style.

* Edit global parameters at the top of each script.
* Run scripts from the repository root.
* RQ1 and RQ2 are standalone controlled experiments.
* RQ3, RQ4, and RQ5 use the real-oracle pipeline.
* TGN and JODIE dependencies are included under `External/`, but their Python package requirements must still be installed in the environment.
* Runtime can vary depending on hardware, especially for TGN and JODIE.

The paper experiments were run on a machine with 128 CPU threads, 1 TiB RAM, and two NVIDIA H200 GPUs.

