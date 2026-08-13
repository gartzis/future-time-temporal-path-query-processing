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

## Additional experiments

This section contains additional experiments that complement the evaluation in the paper. They further analyze the path-ranking step, the relation to the previous single-future-edge setting, the occurrence of paths with multiple future temporal edges, and the behavior of the framework on a larger temporal graph.

### Path-ranking baselines

We compare our framework with four simpler path-processing methods.

For each query, we first run the prediction oracle and obtain the accepted future temporal edges over the future timestamps. All methods use the same observed temporal graph and exactly the same accepted future temporal edges.

The baselines do not rerank a common set of candidate paths. Instead, each method independently constructs candidate temporal paths from the same graph evidence and returns the paths that best satisfy its own ranking objective. Thus, the future-edge evidence is fixed across methods, while the path-processing objective changes.

The methods are:

* **Distance:** ranks candidate temporal paths by path length, with shorter paths ranked first.
* **Time:** ranks candidate temporal paths by formation time, with earlier paths ranked first.
* **Distance-Time:** ranks candidate temporal paths by the temporal shortest-path order, first by path length and then by formation time.
* **Existence:** ranks candidate temporal paths by path-existence probability, with larger probability ranked first.
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
M \in \{\mathrm{PEE}, \mathrm{NDMSE}, \mathrm{NTMSE}\}.
```

Here, k_q is the number of future-valid paths returned up to k. Mean path-quality errors are averaged over covered queries, while Coverage, Path Recall@k, and Path MRR@k follow the definitions of the main evaluation.


<p align="center"><strong>Quality of future-time query answers.</strong></p>


| Oracle / Method | CollegeMsg | Enron | Email-Eu | Bitcoin |
|---|---:|---:|---:|---:|
| **Mean PEE ↓** |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |
| Distance | 0.401 | 0.406 | 0.461 | 0.357 |
| Time | 0.481 | 0.530 | 0.656 | 0.492 |
| Distance-Time | 0.399 | **0.400** | **0.460** | 0.356 |
| Existence | 0.479 | 0.520 | 0.634 | 0.504 |
| Ours | **0.385** | 0.409 | **0.460** | **0.351** |
| **TGN-MaxTime** |  |  |  |  |
| Distance | 0.379 | **0.407** | 0.444 | 0.353 |
| Time | 0.476 | 0.530 | 0.602 | 0.502 |
| Distance-Time | 0.383 | 0.413 | 0.451 | 0.375 |
| Existence | 0.466 | 0.525 | 0.581 | 0.492 |
| Ours | **0.360** | 0.418 | **0.434** | **0.335** |
| **TGN-PerTime** |  |  |  |  |
| Distance | 0.379 | **0.405** | 0.445 | 0.353 |
| Time | 0.484 | 0.552 | 0.597 | 0.472 |
| Distance-Time | 0.380 | 0.407 | 0.442 | 0.353 |
| Existence | 0.466 | 0.525 | 0.581 | 0.492 |
| Ours | **0.361** | 0.418 | **0.429** | **0.324** |
| **JODIE-Frozen** |  |  |  |  |
| Distance | 0.372 | 0.430 | 0.430 | 0.405 |
| Time | 0.481 | 0.542 | 0.656 | 0.468 |
| Distance-Time | 0.359 | 0.426 | 0.421 | 0.344 |
| Existence | 0.481 | 0.542 | 0.656 | 0.467 |
| Ours | **0.320** | **0.393** | **0.418** | **0.308** |
| **JODIE-Update** |  |  |  |  |
| Distance | 0.375 | 0.398 | 0.451 | 0.388 |
| Time | 0.482 | 0.542 | 0.656 | 0.468 |
| Distance-Time | 0.372 | 0.407 | 0.454 | 0.347 |
| Existence | 0.485 | 0.556 | 0.659 | 0.467 |
| Ours | **0.329** | **0.394** | **0.434** | **0.305** |
| **Mean NDMSE ↓** |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |
| Distance | **0.114** | **0.092** | **0.140** | **0.092** |
| Time | 0.227 | 0.243 | 0.345 | 0.252 |
| Distance-Time | **0.114** | **0.092** | **0.140** | **0.092** |
| Existence | 0.227 | 0.233 | 0.339 | 0.304 |
| Ours | 0.146 | 0.118 | 0.153 | 0.133 |
| **TGN-MaxTime** |  |  |  |  |
| Distance | **0.147** | **0.133** | **0.150** | **0.117** |
| Time | 0.244 | 0.229 | 0.344 | 0.260 |
| Distance-Time | **0.147** | **0.133** | **0.150** | **0.117** |
| Existence | 0.234 | 0.212 | 0.333 | 0.283 |
| Ours | 0.155 | 0.140 | 0.166 | 0.142 |
| **TGN-PerTime** |  |  |  |  |
| Distance | **0.147** | **0.132** | **0.149** | **0.117** |
| Time | 0.240 | 0.232 | 0.328 | 0.226 |
| Distance-Time | **0.147** | **0.132** | **0.149** | **0.117** |
| Existence | 0.234 | 0.212 | 0.334 | 0.283 |
| Ours | 0.163 | 0.146 | 0.168 | 0.151 |
| **JODIE-Frozen** |  |  |  |  |
| Distance | **0.083** | 0.084 | **0.094** | 0.079 |
| Time | 0.221 | 0.23588 | 0.34119 | 0.228 |
| Distance-Time | **0.083** | 0.084 | **0.094** | 0.079 |
| Existence | 0.217 | 0.236 | 0.340 | 0.221 |
| Ours | 0.094 | **0.083** | 0.106 | **0.065** |
| **JODIE-Update** |  |  |  |  |
| Distance | **0.099** | **0.094** | **0.117** | 0.075 |
| Time | 0.221 | 0.236 | 0.341 | 0.228 |
| Distance-Time | **0.099** | **0.094** | **0.117** | 0.075 |
| Existence | 0.216 | 0.241 | 0.342 | 0.221 |
| Ours | 0.118 | 0.130 | 0.132 | **0.074** |
| **Mean NTMSE ↓** |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |
| Distance | **0.223** | **0.099** | **0.035** | **0.401** |
| Time | 0.252 | 0.101 | 0.035 | 0.452 |
| Distance-Time | 0.225 | 0.099 | 0.035 | **0.401** |
| Existence | 0.252 | 0.101 | 0.035 | 0.451 |
| Ours | 0.240 | 0.099 | 0.034 | 0.426 |
| **TGN-MaxTime** |  |  |  |  |
| Distance | **0.150** | **0.076** | **0.048** | **0.167** |
| Time | 0.203 | 0.092 | 0.053 | 0.452 |
| Distance-Time | 0.160 | 0.080 | 0.050 | 0.302 |
| Existence | 0.156 | 0.084 | 0.048 | 0.167 |
| Ours | 0.181 | 0.084 | 0.050 | 0.287 |
| **TGN-PerTime** |  |  |  |  |
| Distance | 0.197 | 0.085 | 0.052 | 0.398 |
| Time | 0.226 | 0.094 | 0.053 | 0.452 |
| Distance-Time | 0.206 | 0.088 | 0.052 | 0.401 |
| Existence | **0.156** | **0.084** | **0.048** | **0.167** |
| Ours | 0.209 | 0.089 | 0.051 | 0.328 |
| **JODIE-Frozen** |  |  |  |  |
| Distance | **0.223** | **0.098** | **0.035** | **0.364** |
| Time | 0.252 | 0.101 | 0.035 | 0.452 |
| Distance-Time | 0.232 | 0.100 | 0.035 | 0.420 |
| Existence | 0.250 | 0.100 | 0.035 | 0.45 |
| Ours | 0.226 | 0.099 | 0.034 | 0.398 |
| **JODIE-Update** |  |  |  |  |
| Distance | **0.180** | **0.089** | **0.032** | **0.221** |
| Time | 0.252 | 0.101 | 0.035 | 0.452 |
| Distance-Time | 0.197 | 0.093 | 0.034 | 0.345 |
| Existence | 0.245 | 0.093 | 0.035 | 0.449 |
| Ours | 0.195 | 0.092 | 0.034 | 0.328 |
| **Future-path Coverage ↑** |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |
| Distance | **1.000** | **1.000** | **1.000** | **1.000** |
| Time | **1.000** | **1.000** | **1.000** | **1.000** |
| Distance-Time | **1.000** | **1.000** | **1.000** | **1.000** |
| Existence | **1.000** | **1.000** | **1.000** | **1.000** |
| Ours | **1.000** | **1.000** | **1.000** | **1.000** |
| **TGN-MaxTime** |  |  |  |  |
| Distance | **0.490** | **0.900** | **0.600** | **1.000** |
| Time | **0.490** | **0.900** | **0.600** | **1.000** |
| Distance-Time | **0.490** | **0.900** | **0.600** | **1.000** |
| Existence | **0.490** | **0.900** | **0.600** | **1.000** |
| Ours | 0.440 | 0.880 | 0.590 | 0.990 |
| **TGN-PerTime** |  |  |  |  |
| Distance | **0.490** | **0.900** | **0.600** | **1.000** |
| Time | **0.490** | **0.900** | **0.600** | **1.000** |
| Distance-Time | **0.490** | **0.900** | **0.600** | **1.000** |
| Existence | **0.490** | **0.900** | **0.600** | **1.000** |
| Ours | 0.440 | 0.880 | 0.590 | 0.990 |
| **JODIE-Frozen** |  |  |  |  |
| Distance | **1.000** | **1.000** | **1.000** | **1.000** |
| Time | **1.000** | **1.000** | **1.000** | **1.000** |
| Distance-Time | **1.000** | **1.000** | **1.000** | **1.000** |
| Existence | **1.000** | **1.000** | **1.000** | **1.000** |
| Ours | **1.000** | **1.000** | **1.000** | **1.000** |
| **JODIE-Update** |  |  |  |  |
| Distance | **1.000** | **1.000** | **1.000** | **1.000** |
| Time | **1.000** | **1.000** | **1.000** | **1.000** |
| Distance-Time | **1.000** | **1.000** | **1.000** | **1.000** |
| Existence | **1.000** | **1.000** | **1.000** | **1.000** |
| Ours | **1.000** | **1.000** | **1.000** | **1.000** |

<p align="center"><strong>Top-10 exact-path ranking and shorter-path selection.</strong></p>

| Oracle / Method | CollegeMsg | Enron | Email-Eu | Bitcoin |
|---|---:|---:|---:|---:|
| **Path Recall@10:  ↑** |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |
| Distance | **0.350** | **0.200** | 0.080 | **0.520** |
| Time | 0.000 | 0.020 | 0.000 | 0.000 |
| Distance-Time | **0.350** | **0.200** | 0.080 | **0.520** |
| Existence | 0.000 | 0.010 | 0.000 | 0.010 |
| Ours | 0.230 | 0.110 | **0.120** | 0.320 |
| **TGN-MaxTime** |  |  |  |  |
| Distance | **0.170** | **0.170** | 0.120 | 0.600 |
| Time | 0.000 | 0.010 | 0.000 | 0.000 |
| Distance-Time | 0.150 | 0.140 | 0.100 | 0.290 |
| Existence | 0.010 | 0.020 | 0.020 | 0.080 |
| Ours | 0.090 | 0.100 | **0.130** | **0.650** |
| **TGN-PerTime** |  |  |  |  |
| Distance | **0.170** | **0.170** | 0.120 | 0.600 |
| Time | 0.000 | 0.000 | 0.000 | 0.000 |
| Distance-Time | **0.170** | 0.150 | 0.120 | 0.600 |
| Existence | 0.010 | 0.020 | 0.020 | 0.080 |
| Ours | 0.080 | 0.080 | **0.130** | **0.670** |
| **JODIE-Frozen** |  |  |  |  |
| Distance | **0.310** | **0.060** | 0.050 | **0.150** |
| Time | 0.000 | 0.000 | 0.000 | 0.000 |
| Distance-Time | **0.310** | 0.050 | **0.060** | **0.150** |
| Existence | 0.000 | 0.000 | 0.000 | 0.000 |
| Ours | 0.230 | 0.040 | 0.020 | 0.140 |
| **JODIE-Update** |  |  |  |  |
| Distance | **0.370** | **0.190** | **0.100** | **0.300** |
| Time | 0.000 | 0.000 | 0.000 | 0.000 |
| Distance-Time | 0.340 | 0.150 | 0.070 | 0.280 |
| Existence | 0.000 | 0.000 | 0.000 | 0.000 |
| Ours | 0.260 | 0.070 | 0.030 | 0.210 |
| **Path MRR@10:  ↑** |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |
| Distance | **0.226** | **0.124** | 0.038 | **0.275** |
| Time | 0.000 | 0.006 | 0.000 | 0.000 |
| Distance-Time | **0.226** | **0.124** | 0.038 | **0.275** |
| Existence | 0.000 | 0.005 | 0.000 | 0.002 |
| Ours | 0.153 | 0.090 | **0.038** | 0.154 |
| **TGN-MaxTime** |  |  |  |  |
| Distance | **0.069** | **0.075** | **0.066** | **0.287** |
| Time | 0.000 | 0.002 | 0.000 | 0.000 |
| Distance-Time | 0.065 | 0.067 | 0.053 | 0.129 |
| Existence | 0.010 | 0.015 | 0.015 | 0.065 |
| Ours | 0.031 | 0.030 | 0.056 | 0.197 |
| **TGN-PerTime** |  |  |  |  |
| Distance | **0.069** | **0.075** | **0.066** | **0.287** |
| Time | 0.000 | 0.000 | 0.000 | 0.000 |
| Distance-Time | **0.069** | 0.066 | **0.066** | **0.287** |
| Existence | 0.010 | 0.015 | 0.015 | 0.065 |
| Ours | 0.025 | 0.033 | 0.066 | 0.258 |
| **JODIE-Frozen** |  |  |  |  |
| Distance | 0.207 | **0.053** | 0.036 | 0.125 |
| Time | 0.000 | 0.000 | 0.000 | 0.000 |
| Distance-Time | **0.235** | 0.040 | **0.040** | **0.143** |
| Existence | 0.000 | 0.000 | 0.000 | 0.000 |
| Ours | 0.182 | 0.035 | 0.015 | 0.123 |
| **JODIE-Update** |  |  |  |  |
| Distance | 0.221 | **0.118** | **0.074** | **0.192** |
| Time | 0.000 | 0.000 | 0.000 | 0.000 |
| Distance-Time | **0.222** | 0.057 | 0.044 | 0.185 |
| Existence | 0.000 | 0.000 | 0.000 | 0.000 |
| Ours | 0.190 | 0.044 | 0.025 | 0.138 |
| **Mean shorter-path selection ↓** |  |  |  |  |
| **N2VLP-Static** |  |  |  |  |
| Distance | 0.508 | 0.472 | 0.766 | 0.436 |
| Time | 0.028 | 0.040 | **0.000** | 0.030 |
| Distance-Time | 0.508 | 0.472 | 0.766 | 0.436 |
| Existence | **0.008** | **0.032** | 0.006 | **0.002** |
| Ours | 0.423 | 0.413 | 0.479 | 0.359 |
| **TGN-MaxTime** |  |  |  |  |
| Distance | 0.384 | 0.513 | 0.583 | 0.468 |
| Time | **0.008** | **0.027** | **0.007** | 0.042 |
| Distance-Time | 0.384 | 0.513 | 0.583 | 0.468 |
| Existence | 0.016 | 0.040 | 0.013 | **0.016** |
| Ours | 0.360 | 0.410 | 0.412 | 0.399 |
| **TGN-PerTime** |  |  |  |  |
| Distance | 0.384 | 0.513 | 0.590 | 0.468 |
| Time | **0.008** | 0.058 | 0.017 | 0.044 |
| Distance-Time | 0.384 | 0.513 | 0.590 | 0.468 |
| Existence | 0.016 | **0.040** | **0.013** | **0.016** |
| Ours | 0.371 | 0.429 | 0.422 | 0.395 |
| **JODIE-Frozen** |  |  |  |  |
| Distance | 0.348 | 0.346 | 0.586 | 0.256 |
| Time | **0.016** | **0.030** | **0.006** | **0.028** |
| Distance-Time | 0.348 | 0.346 | 0.586 | 0.256 |
| Existence | 0.032 | 0.042 | 0.008 | 0.040 |
| Ours | 0.447 | 0.437 | 0.689 | 0.304 |
| **JODIE-Update** |  |  |  |  |
| Distance | 0.438 | 0.488 | 0.742 | 0.330 |
| Time | **0.016** | **0.030** | **0.006** | **0.028** |
| Distance-Time | 0.438 | 0.488 | 0.742 | 0.330 |
| Existence | 0.034 | 0.036 | 0.010 | 0.042 |
| Ours | 0.505 | 0.556 | 0.762 | 0.353 |

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

