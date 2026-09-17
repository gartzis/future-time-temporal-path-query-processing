from __future__ import annotations



from pathlib import Path

from typing import Any, Dict, List



import pandas as pd



import oracle_methods

import pipeline_methods



BASE_DIR = Path(__file__).resolve().parent


def repo_path(path: str | Path) -> Path:

    path = Path(path)

    if path.is_absolute():

        return path

    return BASE_DIR / path



DATASET_FILES = {

    "enron": "enron.csv",

    "email_eu": "email-eu.csv",

    "collegemsg": "collegemsg.csv",

    "bitcoin": "bitcoin.csv",

    "dblp_edges": "dblp_edges.csv",

}



QUERY_TEST_FILES = {

    "enron": "enron.tsv",

    "email_eu": "email_eu.tsv",

    "collegemsg": "collegemsg.tsv",

    "bitcoin": "bitcoin.tsv",

    "dblp_edges": "dblp_edges.tsv",

}





def normalize_dataset_name(dataset_name: str) -> str:

    key = dataset_name.lower().replace("-", "_")

    aliases = {

        "email_eu_core": "email_eu",

        "email_eu_core_temporal": "email_eu",

        "college_msg": "collegemsg",

        "college": "collegemsg",

        "bitcoinotc": "bitcoin",

        "bitcoin_otc": "bitcoin",

        "dblp": "dblp_edges",

    }

    return aliases.get(key, key)





def build_dataset_paths(dataset_name: str, data_root: str | Path, query_test_root: str | Path) -> Dict[str, Path | str]:

    dataset_name = normalize_dataset_name(dataset_name)

    if dataset_name not in DATASET_FILES:

        raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset_name not in QUERY_TEST_FILES:

        raise ValueError(f"Unknown query-test dataset: {dataset_name}")

    return {

        "dataset_name": dataset_name,

        "csv_path": Path(data_root) / DATASET_FILES[dataset_name],

        "query_tests_path": Path(query_test_root) / QUERY_TEST_FILES[dataset_name],

    }



DATASET_NAME = "enron"

ORACLE_NAME = "n2vlp_static"



DATA_ROOT = "Data/Datasets"

QUERY_TEST_ROOT = "Data/query_tests"

OUTPUT_ROOT = "Results"



NUM_QUERIES = 100

NUM_RUNS = 10

TOP_K = 10



NUM_LANDMARKS = 5

EDGE_THRESHOLD = 0.5

PATH_EXIST_THRESHOLD = 0.0

SHORTEST_PATH_THRESHOLD = 0.0



USE_CACHE = True

RUN_TIMING_TO_MAX_FUTURE_TIMESTAMP = True

FREEZE_ANSWER_AT_ORIGINAL_FUT_TIME = True



QUERY_SAMPLE_SEED = 42




JODIE_EPOCHS = 50

JODIE_EMBEDDING_DIM = 128

JODIE_LR = 1e-3

JODIE_WEIGHT_DECAY = 1e-5

JODIE_PATH_TOPK_PER_SOURCE = 10

JODIE_EVAL_TOPK_PER_SOURCE = 100

TGN_EDGE_TIME_SELECTION_MODE = "best_score_until_query_time"

MAX_EDGE_TIME_CANDIDATES = None





def build_config() -> Dict[str, Any]:

    return {

        "dataset_name": normalize_dataset_name(DATASET_NAME),

        "oracle_name": ORACLE_NAME.lower(),

        "data_root": repo_path(DATA_ROOT),

        "query_test_root": repo_path(QUERY_TEST_ROOT),

        "output_root": repo_path(OUTPUT_ROOT),

        "num_queries": int(NUM_QUERIES),

        "num_runs": int(NUM_RUNS),

        "top_k": int(TOP_K),

        "num_landmarks": int(NUM_LANDMARKS),

        "edge_threshold": float(EDGE_THRESHOLD),

        "path_exist_threshold": float(PATH_EXIST_THRESHOLD),

        "shortest_path_threshold": float(SHORTEST_PATH_THRESHOLD),

        "use_cache": bool(USE_CACHE),

        "run_timing_to_max_future_timestamp": bool(RUN_TIMING_TO_MAX_FUTURE_TIMESTAMP),

        "freeze_answer_at_original_future_time": bool(FREEZE_ANSWER_AT_ORIGINAL_FUT_TIME),

        "query_sample_seed": QUERY_SAMPLE_SEED,


        "jodie_epochs": int(JODIE_EPOCHS),

        "jodie_embedding_dim": int(JODIE_EMBEDDING_DIM),

        "jodie_lr": float(JODIE_LR),

        "jodie_weight_decay": float(JODIE_WEIGHT_DECAY),

        "jodie_path_topk_per_source": int(JODIE_PATH_TOPK_PER_SOURCE),

        "jodie_eval_topk_per_source": int(JODIE_EVAL_TOPK_PER_SOURCE),

        "tgn_edge_time_selection_mode": str(TGN_EDGE_TIME_SELECTION_MODE),

        "max_edge_time_candidates": MAX_EDGE_TIME_CANDIDATES,

    }





def build_pipeline_config(config: Dict[str, Any], oracle: Dict[str, Any]) -> Any:

    output_dir = Path(config["output_root"]) / config["dataset_name"] / oracle["code_name"]

    cfg = pipeline_methods.PipelineConfig(

        tests_dir=Path(config["query_test_root"]),

        test_suffix="",

        output_dir=output_dir,

        num_random_queries=int(config["num_queries"]),

        num_dataset_iterations=int(config["num_runs"]),

        query_sample_seed=config["query_sample_seed"],

        path_exist_thresholds=(float(config["path_exist_threshold"]),),

        shortest_proba_bounds=(float(config["shortest_path_threshold"]),),

        top_k_result_paths=int(config["top_k"]),

        static_edge_thresholds=(float(config["edge_threshold"]),),

        tgn_edge_thresholds=(float(config["edge_threshold"]),),

        jodie_edge_thresholds=(0.1,),

        use_cache=bool(config["use_cache"]),

        run_timing_to_max_future_timestamp=bool(config["run_timing_to_max_future_timestamp"]),

        freeze_answer_at_original_future_time=bool(config["freeze_answer_at_original_future_time"]),

        jodie_epochs=int(config["jodie_epochs"]),

        jodie_embedding_dim=int(config["jodie_embedding_dim"]),

        jodie_lr=float(config["jodie_lr"]),

        jodie_weight_decay=float(config["jodie_weight_decay"]),

        jodie_path_topk_per_source=int(config["jodie_path_topk_per_source"]),

        jodie_eval_topk_per_source=int(config["jodie_eval_topk_per_source"]),

        tgn_edge_time_selection_mode=str(config["tgn_edge_time_selection_mode"]),

        max_edge_time_candidates=config["max_edge_time_candidates"],

        num_landmarks=int(config["num_landmarks"]),

    )

    if oracle["code_name"] == "tgn_per_time":

        cfg.tgn_edge_time_selection_mode = "current_time"

    if oracle["code_name"] == "tgn_max_time":

        cfg.tgn_edge_time_selection_mode = "best_score_until_query_time"

    return cfg





def print_config(config: Dict[str, Any]) -> None:

    print("Future-time shortest temporal path query processing")

    print("--------------------------------------------------")

    print(f"Dataset: {config['dataset_name']}")

    print(f"Oracle: {config['oracle_name']}")

    print(f"Edge threshold: {config['edge_threshold']}")

    print(f"Path-existence threshold: {config['path_exist_threshold']}")

    print(f"Shortest-path threshold: {config['shortest_path_threshold']}")

    print(f"Cache: {config['use_cache']}")

    print(f"Queries: {config['num_queries']}")

    print(f"Runs: {config['num_runs']}")


    print(f"Data root: {config['data_root']}")

    print(f"Output root: {config['output_root']}")











def write_summary(cfg: Any, detail_rows: List[Dict[str, Any]]) -> None:

    if not detail_rows:

        raise RuntimeError("No result rows were produced. Check dataset paths and query-test files.")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    detail_df = pd.DataFrame(detail_rows)

    summary_df = pipeline_methods.summarize_results(detail_df)

    detail_path = cfg.output_dir / "oracle_temporal_results_detail.tsv"

    summary_path = cfg.output_dir / "oracle_temporal_results_summary.tsv"

    latex_path = cfg.output_dir / "oracle_temporal_results_table.tex"

    detail_df.to_csv(detail_path, sep="\t", index=False)

    summary_df.to_csv(summary_path, sep="\t", index=False)

    latex_path.write_text(pipeline_methods.make_latex_table(summary_df), encoding="utf-8")

    print(f"Saved detail results:      {detail_path}")

    print(f"Saved summary results:     {summary_path}")

    print(f"Saved LaTeX table:        {latex_path}")

    print(summary_df.to_string(index=False))



def _parse_path(value: Any) -> List[int]:

    if pd.isna(value):

        return []

    text = str(value).strip().strip("[]")

    if not text:

        return []

    return [int(float(part.strip())) for part in text.replace("->", ",").split(",") if part.strip()]



def _edit_distance(first: List[int], second: List[int]) -> int:

    previous = list(range(len(second) + 1))

    for index, left in enumerate(first, start=1):

        current = [index]

        for offset, right in enumerate(second, start=1):

            current.append(min(current[-1] + 1, previous[offset] + 1, previous[offset - 1] + (left != right)))

        previous = current

    return previous[-1]



def _read_tsv_files(paths: List[Path]) -> List[pd.DataFrame]:

    frames = []

    for path in paths:

        try:

            frame = pd.read_csv(path, sep="\t")

        except pd.errors.EmptyDataError:

            continue

        if not frame.empty:

            frames.append(frame)

    return frames



def write_oracle_metrics(cfg: Any, oracle: Dict[str, Any], iteration: int) -> None:

    rows = []

    for split, key in (("validation", "validation_metrics"), ("test", "test_metrics")):

        metrics = oracle.get("runtime", {}).get(key)

        if isinstance(metrics, pd.DataFrame) and not metrics.empty:

            frame = metrics.copy()

            frame.insert(0, "iteration", int(iteration))

            frame.insert(0, "split", split)

            frame.insert(0, "oracle", oracle["code_name"])

            rows.append(frame)

    if rows:

        path = cfg.output_dir / "oracle_quality_metrics.tsv"

        frame = pd.concat(rows, ignore_index=True)

        if path.exists():

            frame = pd.concat([pd.read_csv(path, sep="\t"), frame], ignore_index=True)

        frame.to_csv(path, sep="\t", index=False)

    trained = oracle.get("runtime", {}).get("oracle", {})

    if isinstance(trained, dict):

        train_s = float(trained.get("train_s", 0.0))

        if "train_jodie_s" in trained or "train_second_stage_s" in trained:

            train_s = float(trained.get("train_jodie_s", 0.0)) + float(trained.get("train_second_stage_s", 0.0))

        row = pd.DataFrame([{

            "oracle": oracle["code_name"],

            "iteration": int(iteration),

            "training_time_s": train_s,

        }])

        path = cfg.output_dir / "oracle_training_times.tsv"

        if path.exists():

            row = pd.concat([pd.read_csv(path, sep="\t"), row], ignore_index=True)

        row.to_csv(path, sep="\t", index=False)



def write_path_metrics(cfg: Any, detail_rows: List[Dict[str, Any]]) -> None:

    prediction_files = sorted((cfg.output_dir / "path_prediction_results").glob(f"**/*_top{int(cfg.top_k_result_paths)}_predictions.tsv"))

    frames = _read_tsv_files(prediction_files)

    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    detail = pd.DataFrame(detail_rows)

    rows = []

    group_columns = ["dataset", "iteration", "method"]

    for keys, queries in detail.groupby(group_columns, dropna=False):

        selected = predictions

        for column, value in zip(group_columns, keys):

            if not selected.empty:

                selected = selected[selected[column] == value]

        query_keys = queries[["source", "destination", "query_time"]].drop_duplicates()

        path_recall = []

        reciprocal_ranks = []

        covered = []

        pee_values = []

        ndmse_values = []

        ntmse_values = []

        shorter_values = []

        for query in query_keys.itertuples(index=False):

            candidates = selected[

                (selected["source"] == query.source)

                & (selected["destination"] == query.destination)

                & (selected["query_time"] == query.query_time)

            ].sort_values("rank") if not selected.empty else pd.DataFrame()

            valid = candidates[candidates["predictedPath"].fillna("").astype(str).str.len() > 0] if not candidates.empty else candidates

            covered.append(float(not valid.empty))

            if valid.empty:

                path_recall.append(0.0)

                reciprocal_ranks.append(0.0)

                continue

            true_path = _parse_path(valid.iloc[0]["truePath"])

            exact_ranks = [int(row.rank) for row in valid.itertuples() if _parse_path(row.predictedPath) == true_path]

            path_recall.append(float(bool(exact_ranks)))

            reciprocal_ranks.append(1.0 / min(exact_ranks) if exact_ranks else 0.0)

            weights = pd.to_numeric(valid["shortestPathProba"], errors="coerce").fillna(0.0).clip(lower=0.0)

            weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(valid), index=valid.index)

            pee = []

            ndmse = []

            ntmse = []

            shorter = []

            true_edges = max(len(true_path) - 1, 0)

            for row in valid.itertuples():

                predicted_path = _parse_path(row.predictedPath)

                predicted_edges = max(len(predicted_path) - 1, 0)

                pee.append(_edit_distance(true_path, predicted_path) / max(len(true_path), len(predicted_path), 1))

                ndmse.append(((predicted_edges - true_edges) / max(predicted_edges, true_edges, 1)) ** 2)

                true_offset = max(int(row.query_time) - int(row.pivot_time), 0)

                predicted_offset = max(int(row.predictionTime) - int(row.pivot_time), 0)

                ntmse.append(((predicted_offset - true_offset) / max(predicted_offset, true_offset, 1)) ** 2)

                shorter.append(float(predicted_edges < true_edges))

            pee_values.append(float((weights * pee).sum()))

            ndmse_values.append(float((weights * ndmse).sum()))

            ntmse_values.append(float((weights * ntmse).sum()))

            shorter_values.append(float((weights * shorter).sum()))

        rows.append({

            "dataset": keys[0],

            "iteration": int(keys[1]),

            "method": keys[2],

            "weighted_pee_at_10": sum(pee_values) / len(pee_values) if pee_values else 0.0,

            "weighted_ndmse_at_10": sum(ndmse_values) / len(ndmse_values) if ndmse_values else 0.0,

            "weighted_ntmse_at_10": sum(ntmse_values) / len(ntmse_values) if ntmse_values else 0.0,

            "path_recall_at_10": sum(path_recall) / len(path_recall) if path_recall else 0.0,

            "path_mrr_at_10": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,

            "weighted_shorter_at_10": sum(shorter_values) / len(shorter_values) if shorter_values else 0.0,

            "future_coverage": sum(covered) / len(covered) if covered else 0.0,

            "queries": len(query_keys),

        })

    metrics = pd.DataFrame(rows)

    metrics.to_csv(cfg.output_dir / "path_quality_metrics_by_iteration.tsv", sep="\t", index=False)

    metric_columns = [column for column in metrics.columns if column not in group_columns + ["queries"]]

    summary = metrics.groupby(["dataset", "method"], as_index=False)[metric_columns].mean()

    summary.to_csv(cfg.output_dir / "path_quality_metrics_summary.tsv", sep="\t", index=False)



def write_timing_summary(cfg: Any, detail_rows: List[Dict[str, Any]]) -> None:

    files = sorted((cfg.output_dir / "pipeline_timing").glob("**/*_timing.tsv"))

    frames = _read_tsv_files(files)

    if not frames:

        return

    timing = pd.concat(frames, ignore_index=True)

    active = pd.DataFrame(detail_rows)[["dataset", "iteration", "method"]].drop_duplicates()

    timing = timing.merge(active, on=["dataset", "iteration", "method"], how="inner")

    query_columns = ["dataset", "iteration", "method", "source", "destination", "query_time"]

    numeric_columns = [column for column in timing.select_dtypes(include="number").columns if column not in ["iteration", "source", "destination", "pivot_time", "query_time", "t_cur"]]

    per_query = timing.groupby(query_columns, as_index=False)[numeric_columns].sum()

    per_query.to_csv(cfg.output_dir / "pipeline_timing_by_query.tsv", sep="\t", index=False)

    summary = per_query.groupby(["dataset", "iteration", "method"], as_index=False)[numeric_columns].mean()

    summary.to_csv(cfg.output_dir / "pipeline_timing_summary.tsv", sep="\t", index=False)





def run_full_pipeline(config: Dict[str, Any], oracle: Dict[str, Any]) -> None:

    dataset_name = config["dataset_name"]

    if dataset_name not in DATASET_FILES:

        raise ValueError(f"Unknown dataset: {dataset_name}")

    cfg = build_pipeline_config(config, oracle)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("oracle_quality_metrics.tsv", "oracle_training_times.tsv"):

        path = cfg.output_dir / name

        if path.exists():

            path.unlink()

    dataset_csv = Path(config["data_root"]) / DATASET_FILES[dataset_name]

    pipeline_methods.DATASETS = [(str(dataset_csv), -1)]

    pipeline_methods.apply_config_globals(

        cfg,

        oracle_mode=oracle["core_mode"] if oracle["core_mode"] in {"static", "tgn", "jodie_rr"} else "static",

        jodie_state_mode=oracle.get("state_mode", "frozen"),

    )

    print("[RUN]")

    print(f"dataset={dataset_name}")

    print(f"oracle={oracle['code_name']}")

    print(f"output_dir={cfg.output_dir}")

    print(f"query_tests_dir={cfg.tests_dir}")

    data = pipeline_methods.prepare_dataset(dataset_csv, cfg)

    if data is None:

        raise RuntimeError(f"Dataset preparation failed for {dataset_csv}")

    oracle["config"]["pivot_time"] = int(data["pivot_time"])

    shared_oracle = None

    if oracle_methods.train_once_per_dataset(oracle, cfg):

        print(f"Training {oracle['paper_name']} once for {data['file_stem']}")

        shared_oracle = oracle_methods.prepare_oracle(oracle, data, cfg, iteration=1)

        write_oracle_metrics(cfg, shared_oracle, 1)

    all_detail_rows: List[Dict[str, Any]] = []

    for iteration in range(1, int(config["num_runs"]) + 1):

        print(f"Iteration {iteration}/{config['num_runs']} for {data['file_stem']}")

        tests_df_iter = pipeline_methods.sample_iteration_tests(data, cfg, iteration)

        if shared_oracle is None:

            run_oracle = oracle_methods.prepare_oracle(oracle, data, cfg, iteration=iteration)

            write_oracle_metrics(cfg, run_oracle, iteration)

        else:

            run_oracle = shared_oracle

        rows = pipeline_methods.run_with_progress(

            f"{data['file_stem']} iter={iteration} method={run_oracle['paper_name']}",

            lambda run_oracle=run_oracle, tests_df_iter=tests_df_iter, iteration=iteration: oracle_methods.run_oracle_pipeline(

                run_oracle,

                data,

                cfg,

                tests_df_iter,

                iteration,

            ),

        )

        all_detail_rows.extend(rows)

    write_summary(cfg, all_detail_rows)

    write_path_metrics(cfg, all_detail_rows)

    write_timing_summary(cfg, all_detail_rows)





def main() -> None:

    config = build_config()

    print_config(config)

    oracle = oracle_methods.build_oracle(config["oracle_name"], config)

    run_full_pipeline(config, oracle)





if __name__ == "__main__":

    main()
