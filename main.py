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

}



QUERY_TEST_FILES = {

    "enron": "enron.tsv",

    "email_eu": "email_eu.tsv",

    "collegemsg": "collegemsg.tsv",

    "bitcoin": "bitcoin.tsv",

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





def run_full_pipeline(config: Dict[str, Any], oracle: Dict[str, Any]) -> None:

    dataset_name = config["dataset_name"]

    if dataset_name not in DATASET_FILES:

        raise ValueError(f"Unknown dataset: {dataset_name}")

    cfg = build_pipeline_config(config, oracle)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

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

    all_detail_rows: List[Dict[str, Any]] = []

    for iteration in range(1, int(config["num_runs"]) + 1):

        print(f"Iteration {iteration}/{config['num_runs']} for {data['file_stem']}")

        tests_df_iter = pipeline_methods.sample_iteration_tests(data, cfg, iteration)

        if shared_oracle is None:

            run_oracle = oracle_methods.prepare_oracle(oracle, data, cfg, iteration=iteration)

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





def main() -> None:

    config = build_config()

    print_config(config)

    oracle = oracle_methods.build_oracle(config["oracle_name"], config)

    run_full_pipeline(config, oracle)





if __name__ == "__main__":

    main()

