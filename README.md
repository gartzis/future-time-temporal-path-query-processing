# Future-Time Shortest Temporal Path Query Processing under Prediction Uncertainty

Research code for experiment reproduction on future-time shortest temporal path query processing under prediction uncertainty.

The paper studies future-time shortest temporal path queries, where the query is issued at the current timestamp but the answer concerns a future timestamp. The code combines temporal query processing with prediction oracles, constructs candidate future temporal paths, and ranks them by estimated shortest-path probability.

## What is implemented?

The repository supports the main components of the paper:

- temporal graph loading and pivot-based temporal split
- current-state temporal path initialization
- query-local path records
- landmark-based candidate generation
- oracle-based future-edge scoring
- path-state update across future timestamps
- lower-bound pruning
- Luby-Karp based shortest-path probability estimation
- final candidate path ranking
- cache-aware execution
- weighted top-k path metrics
- runtime and cache measurements

## Repository layout

```text
.
├── main.py
├── oracle_methods.py
├── pipeline_methods.py
├── Experiments/
├── Plotting/
├── FuturePathEstimator/
├── Data/
├── External/
└── Results/
```

## Main files

- **`main.py`**  
  Main experiment runner. Edit the global parameters at the top of the file and run `python main.py`. This file runs the selected dataset and selected oracle.

- **`oracle_methods.py`**  
  Function-based oracle methods for N2VLP-Static, TGN-MaxTime, TGN-PerTime, JODIE-Frozen, and JODIE-Update. The code does not use oracle classes.

- **`pipeline_methods.py`**  
  Shared implementation methods used by `main.py` and `oracle_methods.py`. This file contains the extracted training, scoring, candidate generation, path update, ranking, timing, and output functions from the working code.

- **`FuturePathEstimator/`**  
  Luby-Karp and PredictSP code used by the pipeline.

- **`External/`**  
  Vendored TGN and JODIE code.

## Implemented oracle modes

| Code name | Paper-style name | Type | Used for |
|---|---|---|---|
| `candidate_space` | Candidate-space oracle | Controlled oracle | available in `main.py`, while the paper RQ1 script uses the standalone candidate-generation audit |
| `optimal_temporal_edge` | Optimal temporal-edge oracle | Controlled oracle | available in `main.py`, while the paper RQ2 script uses the standalone optimal-edge audit |
| `n2vlp_static` | N2VLP-Static | Edge-scoring oracle | RQ3, RQ4, RQ5 |
| `tgn_max_time` | TGN-MaxTime | Edge-scoring oracle | RQ3, RQ4, RQ5 |
| `tgn_per_time` | TGN-PerTime | Edge-scoring oracle | RQ3, RQ4 |
| `jodie_frozen` | JODIE-Frozen | Target-ranking oracle | RQ3, RQ4, RQ5 |
| `jodie_update` | JODIE-Update | Target-ranking oracle | RQ3, RQ4, RQ5 |

## Requirements

A typical Python environment includes:

- Python 3.9+
- `numpy`
- `pandas`
- `scipy`
- `networkx`
- `scikit-learn`
- `matplotlib`
- `node2vec`
- `torch`

Example setup:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Input format

The temporal edge streams are under `Data/Datasets/`.

Expected files:

```text
Data/Datasets/enron.csv
Data/Datasets/email-eu.csv
Data/Datasets/collegemsg.csv
Data/Datasets/bitcoin.csv
```

Each file should contain source, destination, and timestamp columns. Common variants are accepted:

- source column: `source`, `src`, `u`
- destination column: `destination`, `target`, `dst`, `v`
- time column: `time`, `timestamp`, `ts`, `t`

The query files are under:

```text
Data/query_tests/
```

They use only the dataset name:

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

## Running the code

The code follows a simple research-script style. There are no command-line arguments in `main.py`.

To run a single experiment, edit the global parameters at the top of `main.py`:

```python
DATASET_NAME = "enron"
ORACLE_NAME = "n2vlp_static"
USE_CACHE = True
NUM_QUERIES = 100
NUM_RUNS = 10
```

Then run:

```bash
python main.py
```


## RQ scripts

Each RQ script also uses editable global parameters at the top. They can be launched either from the repository root or from inside `Experiments/`.

```bash
python Experiments/run_RQ1_candidate_space_oracle.py
python Experiments/run_RQ2_optimal_temporal_edge_oracle.py
python Experiments/run_RQ3_oracle_quality.py
python Experiments/run_RQ4_full_pipeline.py
python Experiments/run_RQ5_runtime_cache.py
```

`run_RQ1_candidate_space_oracle.py` runs the standalone candidate-space audit used for the landmark sensitivity experiment. It does not call prediction models and does not route through `main.py`. The full setting uses landmark sizes `[1, 3, 5]`, matching the paper table. By default, the file runs the paper-scale setting. For a quick check, set `MAX_TEST_ROWS_PER_DATASET` to a small value such as 5 and optionally restrict `DATASETS` or `LANDMARK_POOL_SIZES_TO_RUN`.

`run_RQ2_optimal_temporal_edge_oracle.py` runs the standalone optimal temporal-edge oracle experiment. It uses only true future temporal edges at their actual timestamps, varies the controlled edge probability and shortest-path threshold, and writes the prediction files used for the RQ2 heatmaps. By default, the file runs the paper-scale setting. For a quick check, set the row limit to a small value first. In RQ1 use:

```python
MAX_TEST_ROWS_PER_DATASET = 5
```

In RQ2 use:

```python
MAX_TESTS_PER_DATASET = 5
```

Then restore it to `None` for the full run.


## Outputs

Outputs are written under `Results/`.

Depending on the selected experiment, the code writes:

- per-query predictions
- accepted predicted edges
- summary metrics
- weighted top-k metrics
- runtime breakdowns
- cache counters
- heatmap values
- LaTeX table files

## External models

The repository includes external code for:

- TGN under `External/tgn/`
- JODIE under `External/jodie/`

These folders are vendored from their original implementations and are kept separate from the paper-specific query-processing code.

## Reproducibility notes

The selected oracle is changed by editing `ORACLE_NAME` at the top of `main.py`. The main runner performs the full selected pipeline and calls the oracle implementation from `oracle_methods.py`.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{gkartzios_future_time_paths,
  title = {Future-Time Shortest Temporal Path Query Processing under Prediction Uncertainty},
  author = {Gkartzios, Christos and Pitoura, Evaggelia},
  year = {2026}
}
```
