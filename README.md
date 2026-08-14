<div align="center">

<h1>
Future-Time Shortest Temporal Path Query Processing<br>
under Prediction Uncertainty
</h1>

<strong>Research code for reproducing the experiments of the paper.</strong>

<br><br>

<a href="#quick-start">Quick start</a> · <a href="#experiments">Experiments</a> · <a href="#additional-experiments">Additional experiments</a> · <a href="#input-data">Input data</a>

<br><br>

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-supported-ee4c2c)
![scikit-learn](https://img.shields.io/badge/scikit--learn-supported-f7931e)
![NetworkX](https://img.shields.io/badge/NetworkX-supported-376795)
![Temporal Graphs](https://img.shields.io/badge/Temporal%20Graphs-query%20processing-blueviolet)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20644952.svg)](https://doi.org/10.5281/zenodo.20644952)

</div>

---



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


## Additional experiments

This section contains additional experiments that complement the evaluation in the paper. They further analyze the path-ranking step, the relation to the previous single-future-edge setting, the occurrence of paths with multiple future temporal edges, and the behavior of the framework on a larger temporal graph.

### DBLP dataset

To evaluate the framework on a larger temporal graph, we additionally construct a temporal coauthorship network from DBLP. We consider publications from ICDE, SIGMOD, EDBT, VLDB, and PVLDB from 1996 to 2026.

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
| **1** | 0.895 | 0.850 | **1.851** |
| 3 | 0.895 | 0.850 | 2.329 |
| 5 | **0.900** | **0.860** | 2.656 |

Increasing the number of landmarks from one to three provides no improvement in edge and path recall, while increasing runtime. Increasing the number further to five provides only a small improvement in recovery, at a higher query-processing cost. We therefore use one landmark for DBLP in the remaining experiments.

### Baselines

We compare our framework with four simpler path-processing methods.

For each query, we first run the prediction oracle and obtain the accepted future temporal edges over the future timestamps. All methods use the same observed temporal graph and exactly the same accepted future temporal edges.

Each method independently constructs candidate temporal paths from the same graph evidence and returns the paths that best satisfy its own ranking objective. Thus, the future-edge evidence is fixed across methods, while the path-processing objective changes.

The methods are:

* **Distance (DisB):** ranks candidate temporal paths by path length, with shorter paths ranked first.
* **Time (TB):** ranks candidate temporal paths by formation time, with earlier paths ranked first.
* **Distance-Time (DisTB):** ranks candidate temporal paths by the temporal shortest-path order, first by path length and then by formation time.
* **Existence (EB):** ranks candidate temporal paths by path-existence probability, with larger probability ranked first.
* **Ours:** ranks candidate temporal paths by estimated shortest-path probability, which considers both whether a path exists and whether any shorter or earlier candidate path also exists.


For this comparison, we use the same path-quality errors as in the main evaluation, but report their mean over the returned paths instead of their score-weighted versions. This gives a common evaluation across methods whose ranking scores have different meanings.

We report:

- **PEE:** measures the difference between the predicted and true paths using normalized edit distance.
- **NDMSE:** measures the normalized difference between the predicted and true path lengths.
- **NTMSE:** measures the normalized difference between the predicted and true path formation times.
- **Coverage:** measures the fraction of queries for which at least one future-valid path is returned.
- **Path Recall@10:** measures whether the true future path appears among the first 10 returned paths.
- **Path MRR@10:** measures how early the true future path appears in the returned ranking.
-  **Shorter-path selection:** measures the fraction of returned paths that are shorter than the true future path.

For each path-quality metric, we report the mean over the first k future-valid returned paths:

```math
\mathrm{Mean}\ M@k(q)
=
\frac{1}{k_q}
\sum_{i=1}^{k_q}
M(\hat{P}_{q,i}, P_q),
\qquad
M \in \{\mathrm{PEE}, \mathrm{NDMSE}, \mathrm{NTMSE}, \mathrm{Shorter-path-selection}\}.
```

Here, k_q is the number of future-valid paths returned up to k. Mean path-quality errors are averaged over covered queries, while Coverage, Path Recall@k, and Path MRR@k follow the definitions of the main evaluation.


<p align="center"><strong>Quality of future-time query answers.</strong></p>

Oracle / Method | CollegeMsg | Enron | Email-Eu | Bitcoin | DBLP |
|---|---:|---:|---:|---:|---:|
| **Mean PEE ↓** |  |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |  |
| DisB | 0.401 | 0.406 | 0.461 | 0.357 | **0.409** |
| TB | 0.481 | 0.530 | 0.656 | 0.492 | 0.620|
| DisTB | 0.399 | **0.400** | **0.460** | 0.356 | **0.409** |
| EB | 0.479 | 0.520 | 0.634 | 0.504 | 0.633 |
| Ours | **0.385** | 0.409 | **0.460** | **0.351** | 0.428 |
| **TGN-MaxTime** |  |  |  |  |  |
| DisB | 0.379 | **0.407** | 0.444 | 0.353 | **0.485** |
| TB | 0.476 | 0.530 | 0.602 | 0.502 | 0.635 |
| DisTB | 0.383 | 0.413 | 0.451 | 0.375 | **0.485** |
| EB | 0.466 | 0.525 | 0.581 | 0.492 | 0.618 |
| Ours | **0.360** | 0.418 | **0.434** | **0.335** | 0.493 |
| **TGN-PerTime** |  |  |  |  |  |
| DisB | 0.379 | **0.405** | 0.445 | 0.353 | 0.485 |
| TB | 0.484 | 0.552 | 0.597 | 0.472 | 0.629 |
| DisTB | 0.380 | 0.407 | 0.442 | 0.353 | **0.482** |
| EB | 0.466 | 0.525 | 0.581 | 0.492 | 0.618 |
| Ours | **0.361** | 0.418 | **0.429** | **0.324** | 0.494 |
| **JODIE-Frozen** |  |  |  |  |  |
| DisB | 0.372 | 0.430 | 0.430 | 0.405 | **0.418** |
| TB | 0.481 | 0.542 | 0.656 | 0.468 | 0.614 |
| DisTB | 0.359 | 0.426 | 0.421 | 0.344 | 0.419 |
| EB | 0.481 | 0.542 | 0.656 | 0.467 | 0.614 |
| Ours | **0.320** | **0.393** | **0.418** | **0.308** | **0.418** |
| **JODIE-Update** |  |  |  |  |  |
| DisB | 0.375 | 0.398 | 0.451 | 0.388 | **0.422** |
| TB | 0.482 | 0.542 | 0.656 | 0.468 | 0.614 |
| DisTB | 0.372 | 0.407 | 0.454 | 0.347 | **0.422** |
| EB | 0.485 | 0.556 | 0.659 | 0.467 | 0.614 |
| Ours | **0.329** | **0.394** | **0.434** | **0.305** | **0.422** |
| **Mean NDMSE ↓** |  |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |  |
| DisB | **0.114** | **0.092** | **0.140** | **0.092** | **0.112** |
| TB | 0.227 | 0.243 | 0.345 | 0.252 | 0.265 |
| DisTB | **0.114** | **0.092** | **0.140** | **0.092** | **0.112** |
| EB | 0.227 | 0.233 | 0.339 | 0.304 | 0.309 |
| Ours | 0.146 | 0.118 | 0.153 | 0.133 | 0.124 |
| **TGN-MaxTime** |  |  |  |  |  |
| DisB | **0.147** | **0.133** | **0.150** | **0.117** | **0.094** |
| TB | 0.244 | 0.229 | 0.344 | 0.260 | 0.290 |
| DisTB | **0.147** | **0.133** | **0.150** | **0.117** | **0.094** |
| EB | 0.234 | 0.212 | 0.333 | 0.283 | 0.256 |
| Ours | 0.155 | 0.140 | 0.166 | 0.142 | 0.098 |
| **TGN-PerTime** |  |  |  |  |  |
| DisB | **0.147** | **0.132** | **0.149** | **0.117** | **0.094** |
| TB | 0.240 | 0.232 | 0.328 | 0.226 | 0.278 |
| DisTB | **0.147** | **0.132** | **0.149** | **0.117** | **0.094** |
| EB | 0.234 | 0.212 | 0.334 | 0.283 | 0.256 |
| Ours | 0.163 | 0.146 | 0.168 | 0.151 | 0.099 |
| **JODIE-Frozen** |  |  |  |  |  |
| DisB | **0.083** | 0.084 | **0.094** | 0.079 | **0.076** |
| TB | 0.221 | 0.23588 | 0.34119 | 0.228 | 0.254 |
| DisTB | **0.083** | 0.084 | **0.094** | 0.079 | **0.076** |
| EB | 0.217 | 0.236 | 0.340 | 0.221 | 0.254 |
| Ours | 0.094 | **0.083** | 0.106 | **0.065** | 0.083 |
| **JODIE-Update** |  |  |  |  |  |
| DisB | **0.099** | **0.094** | **0.117** | 0.075 | **0.077** |
| TB | 0.221 | 0.236 | 0.341 | 0.228 | 0.254 |
| DisTB | **0.099** | **0.094** | **0.117** | 0.075 | **0.077** |
| EB | 0.216 | 0.241 | 0.342 | 0.221 | 0.254 |
| Ours | 0.118 | 0.130 | 0.132 | **0.074** | 0.086 |
| **Mean NTMSE ↓** |  |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |  |
| DisB | **0.223** | **0.099** | **0.035** | **0.401** | 0.077 |
| TB | 0.252 | 0.101 | 0.035 | 0.452 | 0.083 |
| DisTB | 0.225 | 0.099 | 0.035 | **0.401** | 0.077 |
| EB | 0.252 | 0.101 | 0.035 | 0.451 | **0.072** |
| Ours | 0.240 | 0.099 | 0.034 | 0.426 | 0.075|
| **TGN-MaxTime** |  |  |  |  |  |
| DisB | **0.150** | **0.076** | **0.048** | **0.167** | **0.017** |
| TB | 0.203 | 0.092 | 0.053 | 0.452 | 0.040 |
| DisTB | 0.160 | 0.080 | 0.050 | 0.302 | 0.023 |
| EB | 0.156 | 0.084 | 0.048 | 0.167 | 0.018 |
| Ours | 0.181 | 0.084 | 0.050 | 0.287 | 0.023 |
| **TGN-PerTime** |  |  |  |  |  |
| DisB | 0.197 | 0.085 | 0.052 | 0.398 | 0.042 |
| TB | 0.226 | 0.094 | 0.053 | 0.452 | 0.077 |
| DisTB | 0.206 | 0.088 | 0.052 | 0.401 | 0.062 |
| EB | **0.156** | **0.084** | **0.048** | **0.167** | **0.018** |
| Ours | 0.209 | 0.089 | 0.051 | 0.328 | 0.058 |
| **JODIE-Frozen** |  |  |  |  |  |
| DisB | **0.223** | **0.098** | **0.035** | **0.364** | **0.074** |
| TB | 0.252 | 0.101 | 0.035 | 0.452 | 0.083 |
| DisTB | 0.232 | 0.100 | 0.035 | 0.420 | 0.082|
| EB | 0.250 | 0.100 | 0.035 | 0.45 | 0.083 |
| Ours | 0.226 | 0.099 | 0.034 | 0.398 | 0.081 |
| **JODIE-Update** |  |  |  |  |  |
| DisB | **0.180** | **0.089** | **0.032** | **0.221** | **0.073** |
| TB | 0.252 | 0.101 | 0.035 | 0.452 | 0.083 |
| DisTB | 0.197 | 0.093 | 0.034 | 0.345 | 0.082 |
| EB | 0.245 | 0.093 | 0.035 | 0.449 | 0.083|
| Ours | 0.195 | 0.092 | 0.034 | 0.328 | 0.080 |
| **Future-path Coverage ↑** |  |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |  |
| DisB | **1.000** | **1.000** | **1.000** | **1.000** | **1.000**  |
| TB | **1.000** | **1.000** | **1.000** | **1.000** | **1.000**  |
| DisTB | **1.000** | **1.000** | **1.000** | **1.000** | **1.000**  |
| EB | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Ours | **1.000** | **1.000** | **1.000** | **1.000** | **1.000**  |
| **TGN-MaxTime** |  |  |  |  |  |
| DisB | **0.490** | **0.900** | **0.600** | **1.000** | 0.880 |
| TB | **0.490** | **0.900** | **0.600** | **1.000** | 0.880 |
| DisTB | **0.490** | **0.900** | **0.600** | **1.000** | 0.880 |
| EB | **0.490** | **0.900** | **0.600** | **1.000** | 0.880|
| Ours | 0.440 | 0.880 | 0.590 | 0.990 | 0.880 |
| **TGN-PerTime** |  |  |  |  |  |
| DisB | **0.490** | **0.900** | **0.600** | **1.000** | 0.880 |
| TB | **0.490** | **0.900** | **0.600** | **1.000** | 0.880 |
| DisTB | **0.490** | **0.900** | **0.600** | **1.000** | 0.880 |
| EB | **0.490** | **0.900** | **0.600** | **1.000** | 0.880 |
| Ours | 0.440 | 0.880 | 0.590 | 0.990 | 0.880 |
| **JODIE-Frozen** |  |  |  |  |  |
| DisB | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| TB | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| DisTB | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| EB | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Ours | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| **JODIE-Update** |  |  |  |  |  |
| DisB | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| TB | **1.000** | **1.000** | **1.000** | **1.000** | 1.0 |
| DisTB | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| EB | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Ours | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

<p align="center"><strong>Top-10 exact-path ranking and shorter-path selection.</strong></p>

| Oracle / Method | CollegeMsg | Enron | Email-Eu | Bitcoin | DBLP |
|---|---:|---:|---:|---:|---:|
| **Path Recall@10:  ↑** |  |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |  |
| DisB | **0.350** | **0.200** | 0.080 | **0.520** | 0.1 |
| TB | 0.000 | 0.020 | 0.000 | 0.000 | 0.0 |
| DisTB | **0.350** | **0.200** | 0.080 | **0.520** | 0.1 |
| EB | 0.000 | 0.010 | 0.000 | 0.010 | 0.01 |
| Ours | 0.230 | 0.110 | **0.120** | 0.320 | **0.14** |
| **TGN-MaxTime** |  |  |  |  |  |
| DisB | **0.170** | **0.170** | 0.120 | 0.600 | 0.0 |
| TB | 0.000 | 0.010 | 0.000 | 0.000 | 0.0 |
| DisTB | 0.150 | 0.140 | 0.100 | 0.290 | 0.0 |
| EB | 0.010 | 0.020 | 0.020 | 0.080 | 0.0 |
| Ours | 0.090 | 0.100 | **0.130** | **0.650** | 0.0 |
| **TGN-PerTime** |  |  |  |  |  |
| DisB | **0.170** | **0.170** | 0.120 | 0.600 | 0.0 |
| TB | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| DisTB | **0.170** | 0.150 | 0.120 | 0.600 | 0.0 |
| EB | 0.010 | 0.020 | 0.020 | 0.080 | 0.0 |
| Ours | 0.080 | 0.080 | **0.130** | **0.670** | 0.0 |
| **JODIE-Frozen** |  |  |  |  |  |
| DisB | **0.310** | **0.060** | 0.050 | **0.150** | **0.01** |
| TB | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| DisTB | **0.310** | 0.050 | **0.060** | **0.150** | **0.01** |
| EB | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| Ours | 0.230 | 0.040 | 0.020 | 0.140 | 0.0 |
| **JODIE-Update** |  |  |  |  |  |
| DisB | **0.370** | **0.190** | **0.100** | **0.300** | **0.01** |
| TB | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| DisTB | 0.340 | 0.150 | 0.070 | 0.280 | **0.01** |
| EB | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| Ours | 0.260 | 0.070 | 0.030 | 0.210 | 0.0 |
| **Path MRR@10:  ↑** |  |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |  |
| DisB | **0.226** | **0.124** | 0.038 | **0.275** | 0.050 |
| TB | 0.000 | 0.006 | 0.000 | 0.000 | 0.0 |
| DisTB | **0.226** | **0.124** | 0.038 | **0.275** | 0.050 |
| EB | 0.000 | 0.005 | 0.000 | 0.002 | 0.002 |
| Ours | 0.153 | 0.090 | **0.038** | 0.154 | **0.061** |
| **TGN-MaxTime** |  |  |  |  |  |
| DisB | **0.069** | **0.075** | **0.066** | **0.287** | 0.0 |
| TB | 0.000 | 0.002 | 0.000 | 0.000 | 0.0 |
| DisTB | 0.065 | 0.067 | 0.053 | 0.129 | 0.0 |
| EB | 0.010 | 0.015 | 0.015 | 0.065 | 0.0 |
| Ours | 0.031 | 0.030 | 0.056 | 0.197 | 0.0 |
| **TGN-PerTime** |  |  |  |  |  |
| DisB | **0.069** | **0.075** | **0.066** | **0.287** | 0.0 |
| TB | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| DisTB | **0.069** | 0.066 | **0.066** | **0.287** | 0.0 |
| EB | 0.010 | 0.015 | 0.015 | 0.065 | 0.0 |
| Ours | 0.025 | 0.033 | 0.066 | 0.258 | 0.0 |
| **JODIE-Frozen** |  |  |  |  |  |
| DisB | 0.207 | **0.053** | 0.036 | 0.125 | **0.005** |
| TB | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| DisTB | **0.235** | 0.040 | **0.040** | **0.143** | **0.005** |
| EB | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| Ours | 0.182 | 0.035 | 0.015 | 0.123 | 0.0 |
| **JODIE-Update** |  |  |  |  |  |
| DisB | 0.221 | **0.118** | **0.074** | **0.192** | **0.005** |
| TB | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| DisTB | **0.222** | 0.057 | 0.044 | 0.185 | **0.005** |
| EB | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| Ours | 0.190 | 0.044 | 0.025 | 0.138 | 0.0 |
| **Mean shorter-path selection ↓** |  |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |  |
| DisB | 0.508 | 0.472 | 0.766 | 0.436 | 0.518 |
| TB | 0.028 | 0.040 | **0.000** | 0.030 | 0.036 |
| DisTB | 0.508 | 0.472 | 0.766 | 0.436 | 0.518 |
| EB | **0.008** | **0.032** | 0.006 | **0.002** | **0.004** |
| Ours | 0.423 | 0.413 | 0.479 | 0.359 | 0.444 |
| **TGN-MaxTime** |  |  |  |  |  |
| DisB | 0.384 | 0.513 | 0.583 | 0.468 | 0.169 |
| TB | **0.008** | **0.027** | **0.007** | 0.042 | **0.006** |
| DisTB | 0.384 | 0.513 | 0.583 | 0.468 | 0.169 |
| EB | 0.016 | 0.040 | 0.013 | **0.016** | 0.010 |
| Ours | 0.360 | 0.410 | 0.412 | 0.399 | 0.148 |
| **TGN-PerTime** |  |  |  |  |  |
| DisB | 0.384 | 0.513 | 0.590 | 0.468 | 0.169 |
| TB | **0.008** | 0.058 | 0.017 | 0.044 | 0.008 |
| DisTB | 0.384 | 0.513 | 0.590 | 0.468 | 0.169 |
| EB | 0.016 | **0.040** | **0.013** | **0.016** | **0.010** |
| Ours | 0.371 | 0.429 | 0.422 | 0.395 | 0.143 |
| **JODIE-Frozen** |  |  |  |  |  |
| DisB | 0.348 | 0.346 | 0.586 | 0.256 | 0.430 |
| TB | **0.016** | **0.030** | **0.006** | **0.028** | **0.032** |
| DisTB | 0.348 | 0.346 | 0.586 | 0.256 | 0.43 |
| EB | 0.032 | 0.042 | 0.008 | 0.040 | **0.032** |
| Ours | 0.447 | 0.437 | 0.689 | 0.304 | 0.452 |
| **JODIE-Update** |  |  |  |  |  |
| DisB | 0.438 | 0.488 | 0.742 | 0.330 | 0.436 |
| TB | **0.016** | **0.030** | **0.006** | **0.028** | 0.032 |
| DisTB | 0.438 | 0.488 | 0.742 | 0.330 | 0.436 |
| EB | 0.034 | 0.036 | 0.010 | 0.042 | 0.032 |
| Ours | 0.505 | 0.556 | 0.762 | 0.353 | 0.463 |


### Comparison with the single-future-edge method [10]

Our previous work [10] also studies future-time shortest temporal path queries using prediction oracles. The method first computes shortest temporal paths in the observed graph and then uses a prediction oracle to extend them into the future.

For the link-prediction oracle, the probability of a future edge is predicted without predicting its exact future timestamp. As a result, [10] is restricted to paths with a single future edge, since the temporal order of multiple predicted future edges cannot be determined.

The method considers possible extensions through nodes reachable from the source within the current shortest-path distance and returns the path with the highest predicted probability.

Since [10] is restricted to paths with a single future edge, we first examine how often this restriction holds in our evaluation. We report the percentage of true future paths that contain one or multiple future edges.

<p align="center"><strong>Distribution of future temporal edges in the true future paths.</strong></p>

| Dataset | One future edge | Multiple future edges |
|---|---:|---:|
| CollegeMsg | 95.00% | 5.00% |
| Enron | 60.00% | 40.00% |
| Email-Eu | 83.00% | 17.00% |
| Bitcoin | 99.00% | 1.00% |
| DBLP | 79.00% | 21.00% |
| **All** | **83.20%** | **16.80%** |

The setting in [10] is simpler because the single future edge can only act as the final shortcut to the destination. The method therefore considers paths that already exist in the observed graph and extends them with one predicted edge from a reachable node to the destination. When multiple future edges are allowed, this structure no longer applies, since sequences of predicted edges and their temporal ordering must also be considered.

To compare the two approaches under the setting supported by [10], we restrict the evaluation to queries whose true future path contains a single future edge. We evaluate both methods on the same queries and using the same prediction oracle.

Since [10] returns a single predicted path, we compare the two approaches using the mean path-quality errors and Exact@1.

<p align="center"><strong>Comparison with [10] on single-future-edge queries.</strong></p>

| Dataset | Oracle | Method | Mean PEE ↓ | Mean NDMSE ↓ | Exact@1 ↑ |
|---|---|---|---:|---:|---:|
| **CollegeMsg** | TGN-MaxTime | [10] | **0.168** | **0.080** | **0.484** |
|  |  | Ours | 0.341 | 0.151 | 0.000 |
|  | JODIE-Frozen | [10] | 0.429 | 0.363 | 0.000 |
|  |  | Ours | **0.314** | **0.093** | **0.149** |
| **Enron** | TGN-MaxTime | [10] | **0.267** | 0.168 | **0.217** |
|  |  | Ours | 0.409 | **0.143** | 0.000 |
|  | JODIE-Frozen | [10] | 0.425 | 0.357 | 0.000 |
|  |  | Ours | **0.393** | **0.095** | **0.050** |
| **Email-Eu** | TGN-MaxTime | [10] | 0.421 | 0.185 | **0.145** |
|  |  | Ours | **0.417** | **0.158** | 0.024 |
|  | JODIE-Frozen | [10] | 0.474 | 0.414 | 0.000 |
|  |  | Ours | **0.431** | **0.111** | **0.012** |
| **Bitcoin** | TGN-MaxTime | [10] | **0.254** | 0.149 | **0.364** |
|  |  | Ours | 0.332 | **0.144** | 0.000 |
|  | JODIE-Frozen | [10] | 0.403 | 0.331 | 0.000 |
|  |  | Ours | **0.306** | **0.065** | **0.111** |
| **DBLP** | TGN-MaxTime | [10] | **0.417** | 0.190 | **0.000** |
|  |  | Ours | 0.504 | **0.103** | **0.000** |
|  | JODIE-Frozen | [10] | **0.428** | 0.361 | **0.000** |
|  |  | Ours | **0.428** | **0.090** | **0.000** |
