from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR / "Helpers") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "Helpers"))
if str(BASE_DIR / "FuturePathEstimator") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "FuturePathEstimator"))

import read_Edge_Stream as RES
from path_Prediction_Algorithms import (
    predictBiggestSPDistanceShortestPathAndDistance as PredictSP,
)

PIVOT_QUANTILE = 0.90
EDGE_PROBAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PATH_EXIST_THRESHOLDS = [0.0]
SHORTEST_PROBA_BOUNDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LK_RUNS = 20
KEEP_ONLY_PIVOT_VERTICES = True
TESTS_DIR = BASE_DIR / "Data" / "query_tests"
RESULTS_DIR = BASE_DIR / "Results" / "RQ2_optimal_temporal_edge_oracle"
MAX_TESTS_PER_DATASET = None
DATASET_FILTER = ""


DATASET_FILES = {
    "enron": "enron.csv",
    "email_eu": "email-eu.csv",
    "collegemsg": "collegemsg.csv",
    "bitcoin": "bitcoin.csv",
}

QUERY_TEST_FILES = {
    "enron": "enron.tsv",
    "email_eu": "email_eu.tsv",
    "collegemsg": "collegemsg.tsv",
    "bitcoin": "bitcoin.tsv",
}

DATASETS = ["enron", "email_eu", "collegemsg", "bitcoin"]


def _normalize_dataset_name(name: str) -> str:
    key = Path(str(name)).stem.lower().replace("-", "_")
    aliases = {
        "email_eu_core": "email_eu",
        "email_eu_core_temporal": "email_eu",
        "college_msg": "collegemsg",
        "collegemsg": "collegemsg",
        "college": "collegemsg",
        "bitcoinotc": "bitcoin",
        "bitcoin_otc": "bitcoin",
        "ml_bitcoinotc_disperse_noduplicate_sorted": "bitcoin",
        "ia_enron_employees_timestampzero_noduplicate_sorted": "enron",
        "email_eu_core_temporal_timestampzero_noduplicate_sorted": "email_eu",
        "collegemsg_timestampzero_noduplicate_sorted": "collegemsg",
    }
    return aliases.get(key, key)


def _resolve_dataset_path(dataset_name: str) -> Path:
    key = _normalize_dataset_name(dataset_name)
    if key in DATASET_FILES:
        path = BASE_DIR / "Data" / "Datasets" / DATASET_FILES[key]
        if path.exists():
            return path
    path = Path(dataset_name)
    candidates = [path, BASE_DIR / path, BASE_DIR / "Data" / "Datasets" / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find dataset {dataset_name}")


def _normalize_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if {"u", "i", "ts_str"}.issubset(df.columns):
        df = df.rename(columns={"u": "source", "i": "target", "ts_str": "time"})
    if {"source", "destination", "time"}.issubset(df.columns):
        df = df.rename(columns={"destination": "target"})
    if "target" not in df.columns and "destination" in df.columns:
        df = df.rename(columns={"destination": "target"})
    if "time" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "time"})
    required = {"source", "target", "time"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    df = df[["source", "target", "time"]].copy()
    df["source"] = df["source"].astype(int)
    df["target"] = df["target"].astype(int)
    df["time"] = df["time"].astype(int)
    return df.sort_values("time", kind="mergesort").reset_index(drop=True)


def _auto_pivot_from_unique_timestamps(df: pd.DataFrame) -> int:
    unique_times = sorted(df["time"].dropna().astype(int).unique())
    if not unique_times:
        raise ValueError("Cannot compute pivot because the dataset has no timestamps")
    idx = int(PIVOT_QUANTILE * len(unique_times))
    idx = min(max(idx, 0), len(unique_times) - 1)
    return int(unique_times[idx])


def _build_streams(
    df: pd.DataFrame,
    pivot_time: int,
) -> Tuple[
    List[Tuple[int, int, int, float]],
    List[int],
    Dict[int, List[Tuple[int, int, int, float]]],
    List[int],
    Tuple[int, int],
    Dict[str, int],
]:
    actual_edges = [
        (int(u), int(v), int(t), 1.0)
        for u, v, t in df[["source", "target", "time"]].itertuples(index=False)
    ]
    actual_edges.sort(key=lambda e: e[2])
    vertex_set = set()
    pivot_edges = []
    for u, v, t, p in actual_edges:
        if t <= pivot_time:
            vertex_set.add(u)
            vertex_set.add(v)
            pivot_edges.append((u, v, t, p))
    if not vertex_set:
        raise ValueError(f"No pivot vertices found for pivot_time={pivot_time}")
    future_edges_by_time: Dict[int, List[Tuple[int, int, int, float]]] = defaultdict(list)
    skipped_new_vertex_edges = 0
    for u, v, t, _ in actual_edges:
        if t <= pivot_time:
            continue
        if KEEP_ONLY_PIVOT_VERTICES and (u not in vertex_set or v not in vertex_set):
            skipped_new_vertex_edges += 1
            continue
        future_edges_by_time[int(t)].append((u, v, int(t), 1.0))
    future_times = sorted(future_edges_by_time)
    tmin = int(df["time"].min())
    stats = {
        "actual_edges_total": len(actual_edges),
        "pivot_edges": len(pivot_edges),
        "future_edges_used": sum(len(x) for x in future_edges_by_time.values()),
        "future_edges_skipped_new_vertex": skipped_new_vertex_edges,
        "pivot_vertices": len(vertex_set),
        "future_timestamps_used": len(future_times),
        "unique_timestamps": int(df["time"].nunique()),
    }
    return pivot_edges, sorted(vertex_set), future_edges_by_time, future_times, (tmin - 1, pivot_time), stats


def _parse_paths_tsv(dataset_name: str) -> pd.DataFrame:
    key = _normalize_dataset_name(dataset_name)
    candidates = []
    if key in QUERY_TEST_FILES:
        candidates.append(TESTS_DIR / QUERY_TEST_FILES[key])
    candidates.append(TESTS_DIR / f"{dataset_name}.tsv")
    candidates.append(TESTS_DIR / f"{Path(dataset_name).stem}_Path_tests_top100_temporal_similar.tsv")
    for path_tests in candidates:
        if path_tests.exists():
            df = pd.read_csv(path_tests, sep="\t")
            expected = {"source", "destination", "prev_Path", "future_Path", "time"}
            missing = expected.difference(df.columns)
            if missing:
                raise ValueError(f"{path_tests} is missing columns: {sorted(missing)}")
            df = df.copy()
            df["source"] = df["source"].astype(int)
            df["destination"] = df["destination"].astype(int)
            df["time"] = df["time"].astype(int)
            return df
    raise FileNotFoundError(f"Missing test file for {dataset_name}")


def _prediction_stream_for_time(
    future_edges_by_time: Dict[int, List[Tuple[int, int, int, float]]],
    t_cur: int,
    edge_proba: float,
) -> List[Tuple[int, int, int, float]]:
    return [(u, v, t, edge_proba) for u, v, t, _ in future_edges_by_time.get(t_cur, [])]


def _is_valid_source_destination_path(path_nodes, src, dst):
    if not path_nodes:
        return False
    if int(path_nodes[0]) != int(src):
        return False
    if int(path_nodes[-1]) != int(dst):
        return False
    return True


def _run_one_query(
    src: int,
    dst: int,
    fut_time: int,
    pivot_time: int,
    pivot_edges: List[Tuple[int, int, int, float]],
    vertex_list: List[int],
    future_edges_by_time: Dict[int, List[Tuple[int, int, int, float]]],
    future_times: List[int],
    pivot_interval: Tuple[int, int],
    edge_proba: float,
    existence_bound: float,
    shortest_proba_bound: float,
) -> Tuple[int, List[int], int, float, float] | None:
    if src not in vertex_list or dst not in vertex_list:
        return None
    pivot_edges_src_first = sorted(pivot_edges, key=lambda e: (e[2], e[0] != src))
    (
        sp_pivot,
        L_pivot,
        _,
        actual_pivot,
        dest_paths_pivot,
    ) = RES.computeActualShortestPathAndDistance(
        pivot_edges_src_first,
        src,
        vertex_list,
        pivot_interval,
    )
    pivot_baseline_sp = dict(sp_pivot)
    pivot_entry = pivot_baseline_sp.get(dst, (math.inf, [], 0, 0.0))
    pivot_dist = pivot_entry[0]
    pivot_nodes = pivot_entry[1]
    pivot_tform = pivot_entry[2] if len(pivot_entry) > 2 else pivot_time
    reference_len = len(pivot_nodes) - 1 if pivot_nodes else None
    if pivot_dist != math.inf and pivot_nodes:
        last_best_tuple = (pivot_dist, pivot_tform, pivot_nodes, [], 1.0, 1.0)
        last_valid_wrt_reference = True
    else:
        last_best_tuple = (math.inf, math.inf, [], [], 0.0, 0.0)
        last_valid_wrt_reference = False
    answer_at_fut_time_tuple = last_best_tuple
    answer_at_fut_time_valid = last_valid_wrt_reference
    for t_cur in future_times:
        if t_cur <= pivot_time:
            continue
        if t_cur > fut_time:
            break
        pred_stream = _prediction_stream_for_time(
            future_edges_by_time,
            t_cur,
            edge_proba,
        )
        if not pred_stream:
            continue
        result = PredictSP(
            pred_stream,
            L_pivot,
            sp_pivot,
            actual_pivot,
            src,
            vertex_list,
            (pivot_interval[0], t_cur),
            dest_paths_pivot,
            existence_bound,
            shortest_proba_bound,
            LK_RUNS,
            pivot_time=pivot_time,
            pivot_baseline_sp=pivot_baseline_sp,
        )
        shortest_dict, L_dict, actual_dict, dest_paths_dict, best_sp_dict = result[:5]
        sp_pivot = shortest_dict
        L_pivot = L_dict
        actual_pivot = actual_dict
        dest_paths_pivot = dest_paths_dict
        best_tuple = best_sp_dict.get(dst, (math.inf, math.inf, [], [], 0.0, 0.0))
        d_val, t_form, path_nodes, path_edges, p_sp, p_exist = best_tuple
        cand_len = len(path_nodes) - 1 if path_nodes else math.inf
        if reference_len is None and d_val != math.inf and path_nodes:
            reference_len = cand_len
        valid_wrt_reference = (
            reference_len is None and d_val != math.inf
        ) or (
            reference_len is not None and cand_len <= reference_len
        )
        last_best_tuple = (d_val, t_form, path_nodes, path_edges, p_sp, p_exist)
        last_valid_wrt_reference = valid_wrt_reference
        answer_at_fut_time_tuple = last_best_tuple
        answer_at_fut_time_valid = last_valid_wrt_reference
    d_val, t_form, path_nodes, path_edges, p_sp, p_exist = answer_at_fut_time_tuple
    if d_val == math.inf:
        return None
    if not answer_at_fut_time_valid:
        return None
    path_nodes = list(path_nodes)
    if not _is_valid_source_destination_path(path_nodes, src, dst):
        return None
    if int(t_form) > int(fut_time):
        return None
    return len(path_nodes) - 1, path_nodes, int(t_form), float(p_sp), float(p_exist)


def _active_datasets():
    selected = list(DATASETS)
    if isinstance(selected, str):
        selected = [selected]
    if DATASET_FILTER:
        selected = [name for name in selected if DATASET_FILTER in str(name)]
    return selected


def _active_edge_probas():
    return  list(EDGE_PROBAS)


def _active_shortest_bounds():
    return list(SHORTEST_PROBA_BOUNDS)


def _active_max_tests():
    if MAX_TESTS_PER_DATASET is None:
        return None
    return int(MAX_TESTS_PER_DATASET)


def main() -> None:
    results_dir = Path(RESULTS_DIR)
    if not results_dir.is_absolute():
        results_dir = BASE_DIR / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    print("RQ2 optimal temporal-edge oracle")
    for dataset_name in _active_datasets():
        csv_path = _resolve_dataset_path(str(dataset_name))
        key = _normalize_dataset_name(str(dataset_name))
        print(f"\n=== Dataset: {key} ===")
        print(f"csv_path={csv_path}")
        df = _normalize_df(csv_path)
        pivot_time = _auto_pivot_from_unique_timestamps(df)
        pivot_edges, vertex_list, future_edges_by_time, future_times, pivot_interval, stream_stats = _build_streams(
            df,
            pivot_time,
        )
        print(
            f"pivot={pivot_time} | pivot_vertices={len(vertex_list)} | "
            f"future_timestamps={len(future_times)}"
        )
        print(
            f"actual_edges={stream_stats['actual_edges_total']} | "
            f"pivot_edges={stream_stats['pivot_edges']} | "
            f"future_edges_used={stream_stats['future_edges_used']} | "
            f"future_edges_skipped_new_vertex={stream_stats['future_edges_skipped_new_vertex']}"
        )
        tests_df = _parse_paths_tsv(key)
        tests_total = len(tests_df)
        tests_df = tests_df[tests_df["time"] > pivot_time].copy()
        tests_df = tests_df[
            tests_df["source"].isin(vertex_list) & tests_df["destination"].isin(vertex_list)
        ].copy()
        max_tests = _active_max_tests()
        if max_tests is not None:
            tests_df = tests_df.head(max_tests).copy()
        print(f"tests_total={tests_total} | tests_used={len(tests_df)}")
        for existence_bound in PATH_EXIST_THRESHOLDS:
            for shortest_proba_bound in _active_shortest_bounds():
                for edge_proba in _active_edge_probas():
                    print(
                        f"  edge_proba={edge_proba} | "
                        f"existence_bound={existence_bound} | "
                        f"shortest_proba_bound={shortest_proba_bound}"
                    )
                    predictions = []
                    for row in tests_df[["source", "destination", "time"]].itertuples(index=False):
                        prediction = _run_one_query(
                            src=int(row.source),
                            dst=int(row.destination),
                            fut_time=int(row.time),
                            pivot_time=pivot_time,
                            pivot_edges=pivot_edges,
                            vertex_list=vertex_list,
                            future_edges_by_time=future_edges_by_time,
                            future_times=future_times,
                            pivot_interval=pivot_interval,
                            edge_proba=edge_proba,
                            existence_bound=existence_bound,
                            shortest_proba_bound=shortest_proba_bound,
                        )
                        if prediction is None:
                            continue
                        _, nodes, t_pred, p_sp, p_exist = prediction
                        predictions.append(
                            {
                                "source": int(row.source),
                                "destination": int(row.destination),
                                "predictedPath": ",".join(map(str, nodes)),
                                "predictionTime": t_pred,
                                "shortestPathProba": p_sp,
                                "pathProba": p_exist,
                            }
                        )
                    out_dir = (
                        results_dir
                        / f"edge_proba={edge_proba}"
                        / f"existence_bound={existence_bound}"
                        / f"shortest_proba_bound={shortest_proba_bound}"
                        / "predictions"
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / f"{key}_prediction.tsv"
                    pd.DataFrame(
                        predictions,
                        columns=[
                            "source",
                            "destination",
                            "predictedPath",
                            "predictionTime",
                            "shortestPathProba",
                            "pathProba",
                        ],
                    ).to_csv(out_file, sep="\t", index=False)
                    summary_rows.append(
                        {
                            "dataset": key,
                            "pivot_time": pivot_time,
                            "unique_timestamps": stream_stats["unique_timestamps"],
                            "actual_edges_total": stream_stats["actual_edges_total"],
                            "pivot_edges": stream_stats["pivot_edges"],
                            "future_actual_edges_used": stream_stats["future_edges_used"],
                            "future_actual_edges_skipped_new_vertex": stream_stats["future_edges_skipped_new_vertex"],
                            "tests_total": tests_total,
                            "tests_used": len(tests_df),
                            "edge_proba": edge_proba,
                            "existence_bound": existence_bound,
                            "shortest_proba_bound": shortest_proba_bound,
                            "predictions_found": len(predictions),
                            "output_file": str(out_file),
                        }
                    )
        print(f"Done: {key}")
    summary_file = results_dir / "probability_threshold_test_summary.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_file, sep="\t", index=False)
    print(f"\nSaved summary: {summary_file}")


if __name__ == "__main__":
    main()
