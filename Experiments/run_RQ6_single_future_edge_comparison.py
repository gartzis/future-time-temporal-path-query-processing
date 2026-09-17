from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import main as runner
import oracle_methods
import pipeline_methods

DATASETS = ["enron", "email_eu", "collegemsg", "bitcoin", "dblp_edges"]
ORACLES = ["tgn_max_time", "jodie_frozen"]
DATA_ROOT = "Data/Datasets"
QUERY_TEST_ROOT = "Data/query_tests"
NUM_QUERIES = 100
NUM_RUNS = 10
TOP_K = 10
NUM_LANDMARKS = {
    "enron": 5,
    "email_eu": 5,
    "collegemsg": 1,
    "bitcoin": 1,
    "dblp_edges": 1,
}
EDGE_THRESHOLD = 0.5
PATH_EXIST_THRESHOLD = 0.0
SHORTEST_PATH_THRESHOLD = 0.0
USE_CACHE = True
QUERY_SAMPLE_SEED = 42
OUTPUT_ROOT = Path("Results/RQ6_single_future_edge_comparison")


def parse_path(value) -> List[int]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    return [int(float(token.strip())) for token in text.split(",") if token.strip()]


def first_time_after(
    times: Sequence[int], previous_time: int, query_time: int
) -> Optional[int]:
    for t in times:
        t = int(t)
        if t > int(previous_time) and t <= int(query_time):
            return t
    return None


def reconstruct_temporal_path(
    path_nodes: Sequence[int],
    pair_times: Dict[Tuple[int, int], List[int]],
    query_time: int,
):
    temporal_edges = []
    previous_time = -(10**30)
    for hop, (u, v) in enumerate(zip(path_nodes[:-1], path_nodes[1:])):
        times = pair_times.get((int(u), int(v)), [])
        chosen = first_time_after(
            times=times, previous_time=previous_time, query_time=query_time
        )
        if chosen is None:
            return (
                temporal_edges,
                f"cannot_reconstruct:{u}->{v}:after={previous_time}:query={query_time}",
            )
        temporal_edges.append((int(u), int(v), int(chosen), int(hop)))
        previous_time = int(chosen)
    return (temporal_edges, "ok")


def count_future_temporal_edges(
    path_nodes: Sequence[int],
    pair_times: Dict[Tuple[int, int], List[int]],
    pivot_time: int,
    query_time: int,
):
    temporal_edges, status = reconstruct_temporal_path(
        path_nodes=path_nodes, pair_times=pair_times, query_time=query_time
    )
    if status != "ok":
        return (None, temporal_edges, status)
    future_edges = [edge for edge in temporal_edges if int(edge[2]) > int(pivot_time)]
    if future_edges:
        first_future_hop = int(future_edges[0][3])
        expected_hops = list(range(first_future_hop, len(temporal_edges)))
        actual_hops = [int(edge[3]) for edge in future_edges]
        assert actual_hops == expected_hops
    return (len(future_edges), temporal_edges, "ok")


def sequence_edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    a = list(a)
    b = list(b)
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            substitute_cost = previous[j - 1] + (0 if int(x) == int(y) else 1)
            current.append(min(insert_cost, delete_cost, substitute_cost))
        previous = current
    return int(previous[-1])


def path_edit_error(predicted_path: Sequence[int], true_path: Sequence[int]) -> float:
    predicted_path = list(predicted_path)
    true_path = list(true_path)
    edit_distance = sequence_edit_distance(predicted_path, true_path)
    denominator = max(len(predicted_path), len(true_path), 1)
    return float(edit_distance / denominator)


def normalized_distance_mse(
    predicted_path: Sequence[int], true_path: Sequence[int]
) -> float:
    predicted_len = max(0, len(predicted_path) - 1)
    true_len = max(0, len(true_path) - 1)
    denominator = max(predicted_len, true_len, 1) ** 2
    return float((predicted_len - true_len) ** 2 / denominator)


def compute_ours_comparison_metrics(
    topk_tuples: Sequence[Tuple[Any, ...]],
    true_future_path: Sequence[int],
    pivot_time: int,
    top_k: int,
):
    complete_prefix = list(topk_tuples)[: max(0, int(top_k))]
    future_valid = []
    for item in complete_prefix:
        if len(item) < 3:
            continue
        formation_time = item[1]
        path_nodes = list(item[2])
        try:
            is_future_valid = (
                path_nodes
                and math.isfinite(float(formation_time))
                and (int(formation_time) > int(pivot_time))
            )
        except (TypeError, ValueError, OverflowError):
            is_future_valid = False
        if is_future_valid:
            future_valid.append(item)
    if not future_valid:
        return {
            "future_coverage": 0.0,
            "exact_at_1": 0.0,
            "mean_pee": np.nan,
            "mean_ndmse": np.nan,
        }
    selected = future_valid[0]
    predicted_path = list(selected[2])
    true_path = list(true_future_path)
    pee_values = [path_edit_error(list(item[2]), true_path) for item in future_valid]
    ndmse_values = [
        normalized_distance_mse(list(item[2]), true_path) for item in future_valid
    ]
    return {
        "future_coverage": 1.0,
        "exact_at_1": float(predicted_path == true_path),
        "mean_pee": float(np.mean(pee_values)),
        "mean_ndmse": float(np.mean(ndmse_values)),
    }


def run_ours_on_single_edge_queries(
    dataset: str,
    oracle_label: str,
    prepared_oracle,
    data,
    cfg,
    query_specs: Sequence[Dict[str, Any]],
    iteration: int,
):
    supported_specs = [
        row for row in query_specs if int(row["num_future_temporal_edges"]) == 1
    ]
    eligible_tests = data["eligible_tests_df"].copy().reset_index(drop=True)
    supported_indices = [int(row["query_index"]) for row in supported_specs]
    tests_df = eligible_tests.iloc[supported_indices].copy().reset_index(drop=True)
    original_topk_hook = pipeline_methods._topk_prediction_rows_for_query
    captured_topk: Dict[Tuple[int, int, int], List[List[Tuple[Any, ...]]]] = {}

    def capture_topk(
        dataset, file_stem, iteration, method, row, pivot_time, topk_tuples, max_rank
    ):
        key = (int(row.source), int(row.destination), int(row.time))
        captured_topk.setdefault(key, []).append([tuple(item) for item in topk_tuples])
        return original_topk_hook(
            dataset=dataset,
            file_stem=file_stem,
            iteration=iteration,
            method=method,
            row=row,
            pivot_time=pivot_time,
            topk_tuples=topk_tuples,
            max_rank=max_rank,
        )

    pipeline_methods._topk_prediction_rows_for_query = capture_topk
    try:
        oracle_methods.run_oracle_pipeline(
            prepared_oracle, data, cfg, tests_df, iteration
        )
    finally:
        pipeline_methods._topk_prediction_rows_for_query = original_topk_hook
    capture_offsets: Dict[Tuple[int, int, int], int] = {}
    query_rows = []
    for spec in supported_specs:
        key = (int(spec["source"]), int(spec["destination"]), int(spec["query_time"]))
        offset = capture_offsets.get(key, 0)
        captures = captured_topk.get(key, [])
        if offset >= len(captures):
            raise RuntimeError(
                f"The current pipeline did not emit a top-k result for query {key}."
            )
        topk_tuples = captures[offset]
        capture_offsets[key] = offset + 1
        metrics = compute_ours_comparison_metrics(
            topk_tuples=topk_tuples,
            true_future_path=spec["true_future_path"],
            pivot_time=int(data["pivot_time"]),
            top_k=TOP_K,
        )
        query_rows.append(
            {
                "dataset": dataset,
                "oracle": oracle_label,
                "iteration": int(iteration),
                "method": "Ours",
                "query_index": int(spec["query_index"]),
                "source": int(spec["source"]),
                "destination": int(spec["destination"]),
                "query_time": int(spec["query_time"]),
                **metrics,
            }
        )
    if sum((len(items) for items in captured_topk.values())) != len(supported_specs):
        raise RuntimeError(
            "The number of captured current-framework outputs does not match the single-future-edge workload."
        )
    return query_rows


def previous_predict_single_future_edge(
    prev_path: Sequence[int],
    destination: int,
    query_time: int,
    pivot_time: int,
    future_times: Sequence[int],
    prepared_oracle,
):
    candidates = []
    for source_index in range(max(0, len(prev_path) - 2)):
        source = int(prev_path[source_index])
        path = list(prev_path[: source_index + 1]) + [int(destination)]
        candidates.append(
            {
                "u": source,
                "v": int(destination),
                "path": path,
                "distance": len(path) - 1,
            }
        )
    if not candidates:
        return None
    oracle_name = str(prepared_oracle["code_name"])
    candidate_edges = [
        (row["u"], row["v"], int(query_time), int(row["distance"]))
        for row in candidates
    ]
    score_by_pair = {}
    if oracle_name == "tgn_max_time":
        candidate_times = sorted(
            {
                int(t)
                for t in future_times
                if int(pivot_time) < int(t) <= int(query_time)
            }
        )
        if not candidate_times and int(query_time) > int(pivot_time):
            candidate_times = [int(query_time)]
        for current_time in candidate_times:
            scored = oracle_methods.score_candidates(
                prepared_oracle, candidate_edges, t_cur=current_time
            )
            for u, v, _distance, score in scored:
                pair = (int(u), int(v))
                score_by_pair[pair] = max(
                    float(score), score_by_pair.get(pair, -math.inf)
                )
    elif oracle_name == "jodie_frozen":
        jodie_oracle = prepared_oracle["runtime"]["oracle"]
        targets = sorted((int(v) for v in jodie_oracle["item_to_jodie"]))
        all_candidate_edges = [
            (row["u"], target, int(query_time), int(row["distance"]))
            for row in candidates
            for target in targets
            if target != row["u"]
        ]
        topk_key = "jodie_path_topk_per_source"
        had_topk = topk_key in prepared_oracle["config"]
        old_topk = prepared_oracle["config"].get(topk_key)
        prepared_oracle["config"][topk_key] = 0
        try:
            scored = oracle_methods.score_candidates(
                prepared_oracle, all_candidate_edges, t_cur=int(query_time)
            )
        finally:
            if had_topk:
                prepared_oracle["config"][topk_key] = old_topk
            else:
                prepared_oracle["config"].pop(topk_key, None)
        score_by_pair = {
            (int(u), int(v)): float(score)
            for u, v, _distance, score in scored
            if int(v) == int(destination)
        }
    else:
        raise ValueError(f"Unsupported oracle: {oracle_name}")
    scored_candidates = [
        {**row, "score": score_by_pair[row["u"], row["v"]]}
        for row in candidates
        if (row["u"], row["v"]) in score_by_pair
    ]
    if not scored_candidates:
        return None
    return max(scored_candidates, key=lambda row: float(row["score"]))


def register_dataset(dataset: str) -> None:
    if dataset != "dblp_edges":
        return
    runner.DATASET_FILES.setdefault("dblp_edges", "dblp_edges.csv")
    runner.QUERY_TEST_FILES.setdefault("dblp_edges", "dblp_edges.tsv")


def configure_repository(dataset: str, oracle_name: str, model_output_root: Path):
    register_dataset(dataset)
    runner.DATASET_NAME = dataset
    runner.ORACLE_NAME = oracle_name
    runner.DATA_ROOT = DATA_ROOT
    runner.QUERY_TEST_ROOT = QUERY_TEST_ROOT
    runner.OUTPUT_ROOT = str(model_output_root)
    runner.NUM_QUERIES = NUM_QUERIES
    runner.NUM_RUNS = NUM_RUNS
    runner.TOP_K = TOP_K
    runner.NUM_LANDMARKS = NUM_LANDMARKS[dataset]
    runner.EDGE_THRESHOLD = EDGE_THRESHOLD
    runner.PATH_EXIST_THRESHOLD = PATH_EXIST_THRESHOLD
    runner.SHORTEST_PATH_THRESHOLD = SHORTEST_PATH_THRESHOLD
    runner.USE_CACHE = USE_CACHE
    runner.QUERY_SAMPLE_SEED = QUERY_SAMPLE_SEED
    runner.RUN_TIMING_TO_MAX_FUTURE_TIMESTAMP = True
    runner.FREEZE_ANSWER_AT_ORIGINAL_FUT_TIME = True
    config = runner.build_config()
    oracle = oracle_methods.build_oracle(oracle_name, config)
    cfg = runner.build_pipeline_config(config, oracle)
    if oracle_name == "tgn_max_time":
        cfg.tgn_edge_time_selection_mode = "best_score_until_query_time"
        pipeline_methods.apply_config_globals(cfg, oracle_mode="tgn")
    elif oracle_name == "jodie_frozen":
        pipeline_methods.apply_config_globals(
            cfg, oracle_mode="jodie_rr", jodie_state_mode="frozen"
        )
    else:
        raise ValueError(f"Unsupported oracle: {oracle_name}")
    dataset_csv = Path(config["data_root"]) / runner.DATASET_FILES[dataset]
    data = pipeline_methods.prepare_dataset(dataset_csv, cfg)
    if data is None:
        raise RuntimeError(f"Dataset preparation failed: {dataset}")
    oracle["config"]["pivot_time"] = int(data["pivot_time"])
    return (cfg, data, oracle)


def classify_queries(data):
    rows = []
    pivot_time = int(data["pivot_time"])
    pair_times = data["pair_times"]
    tests = data["eligible_tests_df"].copy().reset_index(drop=True)
    for query_index, row in tests.iterrows():
        query_time = int(row["time"])
        true_path = parse_path(row["future_Path"])
        count, _temporal_edges, status = count_future_temporal_edges(
            path_nodes=true_path,
            pair_times=pair_times,
            pivot_time=pivot_time,
            query_time=query_time,
        )
        if status != "ok":
            raise RuntimeError(f"Could not reconstruct query {query_index}: {status}")
        rows.append(
            {
                "query_index": int(query_index),
                "source": int(row["source"]),
                "destination": int(row["destination"]),
                "query_time": query_time,
                "prev_path": parse_path(row["prev_Path"]),
                "true_future_path": true_path,
                "num_future_temporal_edges": int(count),
            }
        )
    return rows


def previous_metrics(prediction, true_path: Sequence[int]):
    if prediction is None:
        return {
            "future_coverage": 0.0,
            "mean_pee": np.nan,
            "mean_ndmse": np.nan,
            "exact_at_1": 0.0,
        }
    predicted_path = list(prediction["path"])
    return {
        "future_coverage": 1.0,
        "mean_pee": path_edit_error(predicted_path, true_path),
        "mean_ndmse": normalized_distance_mse(predicted_path, true_path),
        "exact_at_1": float(predicted_path == list(true_path)),
    }


def run_dataset(dataset: str, oracle_name: str):
    model_output_root = OUTPUT_ROOT / "models" / oracle_name / dataset
    cfg, data, oracle = configure_repository(dataset, oracle_name, model_output_root)
    oracle_label = str(oracle.get("paper_name", oracle_name))
    query_specs = classify_queries(data)
    if not query_specs:
        raise RuntimeError(f"No eligible queries were found for {dataset}.")
    single_specs = [row for row in query_specs if row["num_future_temporal_edges"] == 1]
    multiple_count = sum((row["num_future_temporal_edges"] > 1 for row in query_specs))
    if len(single_specs) + multiple_count != len(query_specs):
        raise RuntimeError(f"A query for {dataset} has no future temporal edge.")
    distribution = {
        "dataset": dataset,
        "queries": len(query_specs),
        "one_future_edge": len(single_specs),
        "multiple_future_edges": multiple_count,
        "one_future_edge_pct": 100.0 * len(single_specs) / len(query_specs),
        "multiple_future_edges_pct": 100.0 * multiple_count / len(query_specs),
    }
    query_rows = []
    for iteration in range(1, NUM_RUNS + 1):
        prepared_oracle = oracle_methods.prepare_oracle(
            oracle, data, cfg, iteration=iteration
        )
        query_rows.extend(
            run_ours_on_single_edge_queries(
                dataset=dataset,
                oracle_label=oracle_label,
                prepared_oracle=prepared_oracle,
                data=data,
                cfg=cfg,
                query_specs=query_specs,
                iteration=iteration,
            )
        )
        for spec in single_specs:
            prediction = previous_predict_single_future_edge(
                prev_path=spec["prev_path"],
                destination=spec["destination"],
                query_time=spec["query_time"],
                pivot_time=int(data["pivot_time"]),
                future_times=data.get("future_times_test", []),
                prepared_oracle=prepared_oracle,
            )
            query_rows.append(
                {
                    "dataset": dataset,
                    "oracle": oracle_label,
                    "iteration": iteration,
                    "method": "Previous",
                    "query_index": spec["query_index"],
                    "source": spec["source"],
                    "destination": spec["destination"],
                    "query_time": spec["query_time"],
                    **previous_metrics(prediction, spec["true_future_path"]),
                }
            )
    return (query_rows, distribution)


def safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else np.nan


def write_outputs(query_rows, distribution_rows) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    query_df = pd.DataFrame(query_rows)
    query_df.to_csv(
        OUTPUT_ROOT / "single_future_edge_query_metrics.tsv", sep="\t", index=False
    )
    key_columns = ["query_index", "source", "destination", "query_time"]
    for keys, group in query_df.groupby(["dataset", "oracle", "iteration"], sort=False):
        previous_keys = (
            group[group["method"] == "Previous"][key_columns]
            .sort_values(key_columns)
            .reset_index(drop=True)
        )
        ours_keys = (
            group[group["method"] == "Ours"][key_columns]
            .sort_values(key_columns)
            .reset_index(drop=True)
        )
        if not previous_keys.equals(ours_keys):
            raise RuntimeError(f"The methods did not use the same queries for {keys}.")
    iteration_rows = []
    for keys, group in query_df.groupby(
        ["dataset", "oracle", "iteration", "method"], sort=False
    ):
        dataset, oracle_name, iteration, method = keys
        covered = pd.to_numeric(group["future_coverage"], errors="coerce") == 1.0
        iteration_rows.append(
            {
                "dataset": dataset,
                "oracle": oracle_name,
                "iteration": int(iteration),
                "method": method,
                "queries": len(group),
                "covered_queries": int(covered.sum()),
                "mean_pee": safe_mean(group.loc[covered, "mean_pee"]),
                "mean_ndmse": safe_mean(group.loc[covered, "mean_ndmse"]),
                "exact_at_1": safe_mean(group["exact_at_1"]),
            }
        )
    iteration_df = pd.DataFrame(iteration_rows)
    iteration_df.to_csv(
        OUTPUT_ROOT / "single_future_edge_iteration_metrics.tsv", sep="\t", index=False
    )
    summary_rows = []
    for keys, group in iteration_df.groupby(
        ["dataset", "oracle", "method"], sort=False
    ):
        dataset, oracle_name, method = keys
        summary_rows.append(
            {
                "dataset": dataset,
                "oracle": oracle_name,
                "method": method,
                "runs": int(group["iteration"].nunique()),
                "queries": int(round(group["queries"].mean())),
                "mean_pee": safe_mean(group["mean_pee"]),
                "mean_ndmse": safe_mean(group["mean_ndmse"]),
                "exact_at_1": safe_mean(group["exact_at_1"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df["method"] = pd.Categorical(
        summary_df["method"], categories=["Previous", "Ours"], ordered=True
    )
    summary_df = summary_df.sort_values(["dataset", "oracle", "method"]).reset_index(
        drop=True
    )
    summary_df.to_csv(
        OUTPUT_ROOT / "single_future_edge_summary.tsv", sep="\t", index=False
    )
    distribution_df = (
        pd.DataFrame(distribution_rows)
        .drop_duplicates(subset=["dataset"])
        .sort_values("dataset")
    )
    distribution_df.to_csv(
        OUTPUT_ROOT / "future_edge_distribution.tsv", sep="\t", index=False
    )
    print(summary_df.round(4).to_string(index=False))
    print(distribution_df.round(2).to_string(index=False))


def main() -> None:
    all_query_rows = []
    distribution_rows = []
    for dataset in DATASETS:
        for oracle_name in ORACLES:
            query_rows, distribution = run_dataset(dataset, oracle_name)
            all_query_rows.extend(query_rows)
            distribution_rows.append(distribution)
            write_outputs(all_query_rows, distribution_rows)


if __name__ == "__main__":
    main()
