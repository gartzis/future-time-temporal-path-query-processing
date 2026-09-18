<div align="center">

<h1>
Future-Time Shortest Temporal Path Query Processing<br>
under Prediction Uncertainty
</h1>

<strong>Research code for reproducing the experiments of the paper.</strong>

<br><br>

<a href="#quick-start">Quick start</a> · <a href="#experiments">Experiments</a> · <a href="#input-data">Input data</a> · <a href="#complete-results-and-dataset-details">Complete results</a> 

<br><br>

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-supported-ee4c2c)
![scikit-learn](https://img.shields.io/badge/scikit--learn-supported-f7931e)
![NetworkX](https://img.shields.io/badge/NetworkX-supported-376795)
![Temporal Graphs](https://img.shields.io/badge/Temporal%20Graphs-query%20processing-blueviolet)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20644952.svg)](https://doi.org/10.5281/zenodo.20644952)

</div>

---



The paper studies future-time shortest temporal path queries. A query is issued at the current timestamp, but the answer concerns a future timestamp and may depend on edges that have not yet appeared. The code combines temporal query processing with prediction oracles, constructs candidate temporal paths with multiple future edges at different timestamps, and ranks them by estimated shortest-path probability.

## Repository overview

The repository contains:

* controlled-oracle experiments for candidate-space recovery and probability-based path selection
* real-oracle experiments with N2VLP-Static, TGN, and JODIE
* future-time temporal path construction and ranking
* overlap-aware shortest-path probability estimation
* cache-aware query processing
* component evaluation of probability-based path ranking
* comparison with the previous single-future-edge method
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
└── Results/        # generated when experiments are run
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

The paper uses six research questions. RQ1 to RQ5 each have one main runner, while RQ6 is reproduced by two scripts.

```bash
python Experiments/run_RQ1_candidate_space_oracle.py
python Experiments/run_RQ2_optimal_temporal_edge_oracle.py
python Experiments/run_RQ3_oracle_quality.py
python Experiments/run_RQ4_full_pipeline.py
python Experiments/run_RQ5_runtime_cache.py
python Experiments/run_RQ6_path_processing_baselines.py
python Experiments/run_RQ6_single_future_edge_comparison.py
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

### RQ6: Component evaluation and significance of multiple future edges

RQ6 uses two scripts.

```bash
python Experiments/run_RQ6_path_processing_baselines.py
python Experiments/run_RQ6_single_future_edge_comparison.py
```

`run_RQ6_path_processing_baselines.py` evaluates the probability-based path-ranking components while keeping the accepted future temporal edges fixed within each oracle setting. The paper compares:

* **NoProb** (`DisTB` in the script): ignores accepted-edge probabilities and ranks by path length, breaking ties by earlier formation time.
* **NoSP** (`EB` in the script): ranks by path-existence probability.
* **MultiE** (`Ours` in the script): ranks by estimated shortest-path probability.

The script runs all five real-oracle settings. The paper reports TGN-MaxTime and JODIE-Frozen for compactness; the complete results for all five oracles are included below.

`run_RQ6_single_future_edge_comparison.py` first measures how often the true future path contains one or multiple future edges. It then compares MultiE with the previous single-future-edge method on the common subset of queries whose true future path contains exactly one future edge, using TGN-MaxTime and JODIE-Frozen.




## Real oracle names

The real-oracle pipeline supports the following oracle names:

| Code name      | Paper name   | Oracle type           | Temporal use                                                                        |
| -------------- | ------------ | --------------------- | ----------------------------------------------------------------------------------- |
| `n2vlp_static` | N2VLP-Static | Edge-scoring oracle   | Static. Scores node pairs using Node2Vec embeddings from the observed prefix graph. |
| `tgn_max_time` | TGN-MaxTime  | Edge-scoring oracle   | Temporal. Scores future timestamps and keeps the maximum-score timestamp.           |
| `tgn_per_time` | TGN-PerTime  | Edge-scoring oracle   | Temporal. Scores candidate edges at each rollout timestamp.                         |
| `jodie_frozen` | JODIE-Frozen | Target-ranking oracle | Temporal model with fixed observed-prefix state during rollout.                     |
| `jodie_update` | JODIE-Update | Target-ranking oracle | Temporal model with query-local state updates during rollout.                       |

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
Data/Datasets/dblp_edges.csv
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
Data/query_tests/dblp_edges.tsv
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
* RQ6 component-comparison summaries
* future-edge-distribution and SingleE-comparison summaries
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
* RQ3, RQ4, RQ5, and RQ6 use real prediction oracles or outputs from the real-oracle pipeline.
* TGN and JODIE dependencies are included under `External/`, but their Python package requirements must still be installed in the environment.
* Runtime can vary depending on hardware, especially for TGN and JODIE.

The paper experiments were run on a machine with 128 CPU threads, 1 TiB RAM, and two NVIDIA H200 GPUs.


## Complete results and dataset details
[DBLP details](#dblp-dataset-details) · [RQ6 component evaluation](#rq6-component-evaluation) · [Single-future-edge comparison](#comparison-with-the-previous-single-future-edge-method)

This section collects DBLP details and the complete RQ6 results. DBLP is part of the main five-dataset evaluation in the paper. The paper shows a compact subset of the RQ6 component results, while the repository provides the complete results for all five real-oracle settings.

### DBLP dataset details

DBLP is a temporal coauthorship network constructed from publications at ICDE, SIGMOD, EDBT, VLDB, and PVLDB from 1996 to 2026.

- **Nodes:** authors.
- **Temporal edges:** a temporal edge $(u,v,t)$ represents a coauthorship between authors $u$ and $v$ in year $t$.

As in the main evaluation, we use the timestamp at the 90% position as the pivot timestamp $t_p$. The characteristics of the DBLP dataset are shown below.

<p align="center"><strong>Characteristics of the DBLP dataset.</strong></p>

| Dataset | Pivot tₚ | Nodes | Temporal edges | Temporal edges ≤ tₚ | Temporal edges > tₚ | Future timestamps |
|---|---:|---:|---:|---:|---:|---:|
| DBLP | 2023 | 25,954 | 133,850 | 104,191 | 29,659 | 3 |

We also examine the effect of the number of landmarks on DBLP using the same candidate-generation evaluation as in the paper.

<p align="center"><strong>Effect of the number of landmarks on DBLP candidate recovery and query-processing cost.</strong></p>

| Landmarks | Edge Recall ↑ | Path Recall ↑ | Runtime (s) ↓ |
|---:|---:|---:|---:|
| **1** | 0.895 | 0.850 | 1.851 |
| 3 | 0.895 | 0.850 | 2.329 |
| 5 | 0.900 | 0.860 | 2.656 |

Increasing the number of landmarks from one to three provides no improvement in edge and path recall, while increasing runtime. Increasing the number further to five provides only a small improvement in recovery, at a higher query-processing cost. We therefore use one landmark for DBLP in the remaining experiments.

### RQ6 component evaluation

The component evaluation keeps the observed temporal graph and accepted future temporal edges fixed within each oracle setting. It compares:

* **NoProb** (`DisTB` in the script): does not use accepted-edge probabilities and ranks by path length, breaking ties by earlier formation time.
* **NoSP** (`EB` in the script): ranks by path-existence probability.
* **MultiE** (`Ours` in the script): ranks by estimated shortest-path probability.

The paper reports TGN-MaxTime and JODIE-Frozen for compactness. The script `Experiments/run_RQ6_path_processing_baselines.py` runs all five real-oracle settings. The complete Mean PEE@10 and Mean NDMSE@10 results are shown below.

<p align="center"><strong>Mean path edit error@10 ↓</strong></p>

| Oracle / Method | CollegeMsg | Enron | Email-Eu | Bitcoin | DBLP |
|---|---:|---:|---:|---:|---:|
| **N2VLP-Static** |  |  |  |  |  |
| NoProb | 0.399 | 0.400 | 0.460 | 0.356 | 0.409 |
| NoSP | 0.479 | 0.520 | 0.634 | 0.504 | 0.633 |
| MultiE | 0.385 | 0.409 | 0.460 | 0.351 | 0.428 |
| **TGN-MaxTime** |  |  |  |  |  |
| NoProb | 0.383 | 0.413 | 0.451 | 0.375 | 0.485 |
| NoSP | 0.466 | 0.525 | 0.581 | 0.492 | 0.618 |
| MultiE | 0.360 | 0.418 | 0.434 | 0.335 | 0.493 |
| **TGN-PerTime** |  |  |  |  |  |
| NoProb | 0.380 | 0.407 | 0.442 | 0.353 | 0.482 |
| NoSP | 0.466 | 0.525 | 0.581 | 0.492 | 0.618 |
| MultiE | 0.361 | 0.418 | 0.429 | 0.324 | 0.494 |
| **JODIE-Frozen** |  |  |  |  |  |
| NoProb | 0.359 | 0.426 | 0.421 | 0.344 | 0.419 |
| NoSP | 0.481 | 0.542 | 0.656 | 0.467 | 0.614 |
| MultiE | 0.320 | 0.393 | 0.418 | 0.308 | 0.418 |
| **JODIE-Update** |  |  |  |  |  |
| NoProb | 0.372 | 0.407 | 0.454 | 0.347 | 0.422 |
| NoSP | 0.485 | 0.556 | 0.659 | 0.467 | 0.614 |
| MultiE | 0.329 | 0.394 | 0.434 | 0.305 | 0.422 |

<p align="center"><strong>Mean normalized distance MSE@10 ↓</strong></p>

| Oracle / Method | CollegeMsg | Enron | Email-Eu | Bitcoin | DBLP |
|---|---:|---:|---:|---:|---:|
| **N2VLP-Static** |  |  |  |  |  |
| NoProb | 0.114 | 0.092 | 0.140 | 0.092 | 0.112 |
| NoSP | 0.227 | 0.233 | 0.339 | 0.304 | 0.309 |
| MultiE | 0.146 | 0.118 | 0.153 | 0.133 | 0.124 |
| **TGN-MaxTime** |  |  |  |  |  |
| NoProb | 0.147 | 0.133 | 0.150 | 0.117 | 0.094 |
| NoSP | 0.234 | 0.212 | 0.333 | 0.283 | 0.256 |
| MultiE | 0.155 | 0.140 | 0.166 | 0.142 | 0.098 |
| **TGN-PerTime** |  |  |  |  |  |
| NoProb | 0.147 | 0.132 | 0.149 | 0.117 | 0.094 |
| NoSP | 0.234 | 0.212 | 0.334 | 0.283 | 0.256 |
| MultiE | 0.163 | 0.146 | 0.168 | 0.151 | 0.099 |
| **JODIE-Frozen** |  |  |  |  |  |
| NoProb | 0.083 | 0.084 | 0.094 | 0.079 | 0.076 |
| NoSP | 0.217 | 0.236 | 0.340 | 0.221 | 0.254 |
| MultiE | 0.094 | 0.083 | 0.106 | 0.065 | 0.083 |
| **JODIE-Update** |  |  |  |  |  |
| NoProb | 0.099 | 0.094 | 0.117 | 0.075 | 0.077 |
| NoSP | 0.216 | 0.241 | 0.342 | 0.221 | 0.254 |
| MultiE | 0.118 | 0.130 | 0.132 | 0.074 | 0.086 |

Across all 25 oracle-dataset combinations, MultiE has lower Mean PEE@10 and Mean NDMSE@10 than NoSP. The comparison with NoProb is mixed, as discussed in the paper: MultiE is better on Mean PEE@10 in most cases, while the length-based NoProb ranking often has lower Mean NDMSE@10.

### Comparison with the previous single-future-edge method

The previous method, referred to as **SingleE** in the paper, also studies future-time shortest temporal path queries using prediction oracles. It first computes shortest temporal paths in the observed graph and extends them with one predicted future edge. The prediction oracle estimates the probability of that edge without predicting its exact future timestamp. Therefore, SingleE cannot determine the temporal order of multiple predicted future edges.

We first measure how often the true future path contains one or multiple future edges.

<p align="center"><strong>Distribution of future temporal edges in the true future paths.</strong></p>

| Dataset | One future edge | Multiple future edges |
|---|---:|---:|
| CollegeMsg | 95.00% | 5.00% |
| Enron | 60.00% | 40.00% |
| Email-Eu | 83.00% | 17.00% |
| Bitcoin | 99.00% | 1.00% |
| DBLP | 79.00% | 21.00% |

To compare the two methods within the setting supported by SingleE, we restrict the evaluation to queries whose true future path contains exactly one future edge. Both methods use the same queries and prediction oracle. Since SingleE returns one predicted path, we use the first path returned by MultiE and compare Mean PEE and Mean NDMSE.

<p align="center"><strong>Comparison on single-future-edge queries.</strong></p>

| Dataset | Oracle | Method | Mean PEE ↓ | Mean NDMSE ↓ |
|---|---|---|---:|---:|
| **CollegeMsg** | TGN-MaxTime | SingleE | 0.168 | 0.080 |
|  |  | MultiE | 0.341 | 0.151 |
|  | JODIE-Frozen | SingleE | 0.204 | 0.102 |
|  |  | MultiE | 0.314 | 0.093 |
| **Enron** | TGN-MaxTime | SingleE | 0.267 | 0.168 |
|  |  | MultiE | 0.409 | 0.143 |
|  | JODIE-Frozen | SingleE | 0.266 | 0.177 |
|  |  | MultiE | 0.398 | 0.096 |
| **Email-Eu** | TGN-MaxTime | SingleE | 0.421 | 0.185 |
|  |  | MultiE | 0.417 | 0.158 |
|  | JODIE-Frozen | SingleE | 0.441 | 0.221 |
|  |  | MultiE | 0.431 | 0.111 |
| **Bitcoin** | TGN-MaxTime | SingleE | 0.254 | 0.149 |
|  |  | MultiE | 0.332 | 0.144 |
|  | JODIE-Frozen | SingleE | 0.325 | 0.188 |
|  |  | MultiE | 0.306 | 0.065 |
| **DBLP** | TGN-MaxTime | SingleE | 0.417 | 0.190 |
|  |  | MultiE | 0.504 | 0.103 |
|  | JODIE-Frozen | SingleE | 0.418 | 0.225 |
|  |  | MultiE | 0.428 | 0.090 |

SingleE achieves lower Mean PEE in seven of the ten oracle-dataset cases, while MultiE achieves lower Mean NDMSE in nine of the ten cases. MultiE also supports the queries whose true future path requires multiple future edges at different timestamps.
