from __future__ import annotations



from pathlib import Path

from typing import Any, Dict





ORACLE_INFO: Dict[str, Dict[str, str]] = {

    "candidate_space": {

        "paper_name": "Candidate-space oracle",

        "type": "controlled",

        "core_mode": "candidate_space",

        "method_label": "Candidate-Space",

    },

    "optimal_temporal_edge": {

        "paper_name": "Optimal temporal-edge oracle",

        "type": "controlled",

        "core_mode": "optimal_temporal_edge",

        "method_label": "Optimal-Temporal-Edge",

    },

    "n2vlp_static": {

        "paper_name": "N2VLP-Static",

        "type": "edge_scoring",

        "core_mode": "static",

        "method_label": "N2VLP-Static",

    },

    "tgn_max_time": {

        "paper_name": "TGN-MaxTime",

        "type": "edge_scoring",

        "core_mode": "tgn",

        "method_label": "TGN-MaxTime",

    },

    "tgn_per_time": {

        "paper_name": "TGN-PerTime",

        "type": "edge_scoring",

        "core_mode": "tgn",

        "method_label": "TGN-PerTime",

    },

    "jodie_frozen": {

        "paper_name": "JODIE-Frozen",

        "type": "target_ranking",

        "core_mode": "jodie_rr",

        "method_label": "Jodie-Frozen",

    },

    "jodie_update": {

        "paper_name": "JODIE-Update",

        "type": "target_ranking",

        "core_mode": "jodie_rr",

        "method_label": "Jodie-Update",

    },

}





def core():

    import pipeline_methods



    return pipeline_methods





def build_oracle(oracle_name: str, config: Dict[str, Any] | None = None) -> Dict[str, Any]:

    key = oracle_name.lower()

    if key not in ORACLE_INFO:

        raise ValueError(f"Unknown oracle: {oracle_name}")

    oracle = dict(ORACLE_INFO[key])

    oracle["code_name"] = key

    oracle["config"] = config or {}

    oracle["runtime"] = {}

    if key == "tgn_max_time":

        oracle["edge_time_selection_mode"] = "best_score_until_query_time"

    if key == "tgn_per_time":

        oracle["edge_time_selection_mode"] = "current_time"

    if key == "jodie_frozen":

        oracle["state_mode"] = "frozen"

    if key == "jodie_update":

        oracle["state_mode"] = "self_update"

    return oracle





def train_once_per_dataset(oracle: Dict[str, Any], cfg: Any) -> bool:

    name = oracle["code_name"]

    if name == "n2vlp_static":

        return bool(getattr(cfg, "train_lp_once_per_dataset", True))

    if name in {"tgn_max_time", "tgn_per_time"}:

        return bool(getattr(cfg, "train_tgn_once_per_dataset", True))

    if name in {"jodie_frozen", "jodie_update"}:

        return bool(getattr(cfg, "train_jodie_once_per_dataset", True))

    return False





def prepare_oracle(oracle: Dict[str, Any], data: Dict[str, Any], cfg: Any, iteration: int = 1) -> Dict[str, Any]:

    name = oracle["code_name"]

    oracle["runtime"] = {}

    if name == "candidate_space":

        return oracle

    if name == "optimal_temporal_edge":

        oracle["runtime"]["true_future_edges"] = {

            (int(u), int(v), int(t)) for (u, v, t, _p) in data.get("edges_after_90", [])

        }

        return oracle

    if name == "n2vlp_static":

        return prepare_n2vlp_static(oracle, data, cfg, iteration)

    if name in {"tgn_max_time", "tgn_per_time"}:

        return prepare_tgn(oracle, data, cfg, iteration)

    if name in {"jodie_frozen", "jodie_update"}:

        return prepare_jodie(oracle, data, cfg, iteration)

    raise ValueError(f"Unknown oracle: {name}")





def prepare_n2vlp_static(oracle: Dict[str, Any], data: Dict[str, Any], cfg: Any, iteration: int) -> Dict[str, Any]:

    c = core()

    c.apply_config_globals(cfg, oracle_mode="static")

    emb_dir = Path(cfg.output_dir) / "embeddings_static" / f"{data['file_stem']}_{iteration}"

    emb_dir.mkdir(parents=True, exist_ok=True)

    trained_oracle, embeddings_90, threshold, val_df, test_df = c.train_static_oracle(

        data=data,

        cfg=cfg,

        iteration=iteration,

        embeddings_dir=emb_dir,

    )

    oracle["runtime"].update({

        "oracle": trained_oracle,

        "embeddings_90": embeddings_90,

        "threshold": float(threshold),

        "validation_metrics": val_df,

        "test_metrics": test_df,

    })

    return oracle





def prepare_tgn(oracle: Dict[str, Any], data: Dict[str, Any], cfg: Any, iteration: int) -> Dict[str, Any]:

    c = core()

    cfg.tgn_edge_time_selection_mode = oracle.get("edge_time_selection_mode", "best_score_until_query_time")

    c.apply_config_globals(cfg, oracle_mode="tgn")

    trained_oracle, threshold, val_df, test_df = c.train_tgn_oracle(data=data, cfg=cfg, iteration=iteration)

    oracle["runtime"].update({

        "oracle": trained_oracle,

        "threshold": float(threshold),

        "validation_metrics": val_df,

        "test_metrics": test_df,

    })

    return oracle





def prepare_jodie(oracle: Dict[str, Any], data: Dict[str, Any], cfg: Any, iteration: int) -> Dict[str, Any]:

    c = core()

    c.apply_config_globals(cfg, oracle_mode="jodie_rr", jodie_state_mode=oracle.get("state_mode", "frozen"))

    trained_oracle, threshold, val_df, test_df = c.train_jodie_oracle(data=data, cfg=cfg, iteration=iteration)

    oracle["runtime"].update({

        "oracle": trained_oracle,

        "threshold": float(threshold),

        "validation_metrics": val_df,

        "test_metrics": test_df,

    })

    return oracle





def score_candidates(oracle: Dict[str, Any], candidate_edges, query_state=None, t_cur=None, query=None):

    name = oracle["code_name"]

    if name == "candidate_space":

        return score_candidate_space(oracle, candidate_edges, t_cur)

    if name == "optimal_temporal_edge":

        return score_optimal_temporal_edges(oracle, candidate_edges, t_cur)

    if name == "n2vlp_static":

        return score_n2vlp_static(oracle, candidate_edges, t_cur)

    if name in {"tgn_max_time", "tgn_per_time"}:

        return score_tgn(oracle, candidate_edges, t_cur)

    if name in {"jodie_frozen", "jodie_update"}:

        return score_jodie(oracle, candidate_edges, t_cur)

    raise ValueError(f"Unknown oracle: {name}")





def score_candidate_space(oracle: Dict[str, Any], candidate_edges, t_cur=None):

    score = float(oracle["config"].get("candidate_space_score", 1.0))

    accepted = []

    for edge in candidate_edges:

        u, v = int(edge[0]), int(edge[1])

        t = int(edge[2]) if len(edge) > 2 else int(t_cur)

        accepted.append((u, v, t, score))

    return accepted





def score_optimal_temporal_edges(oracle: Dict[str, Any], candidate_edges, t_cur=None):

    score = float(oracle["config"].get("optimal_edge_score", 1.0))

    true_edges = oracle["runtime"].get("true_future_edges", set())

    accepted = []

    for edge in candidate_edges:

        u, v = int(edge[0]), int(edge[1])

        t = int(edge[2]) if len(edge) > 2 else int(t_cur)

        if (u, v, t) in true_edges:

            accepted.append((u, v, t, score))

    return accepted





def score_n2vlp_static(oracle: Dict[str, Any], candidate_edges, t_cur=None):

    c = core()

    rows = []

    for edge in candidate_edges:

        u, v = int(edge[0]), int(edge[1])

        new_len = int(edge[3]) if len(edge) > 3 else 1

        rows.append((u, v, new_len))

    return c._score_candidate_pairs_batch(

        oracle=oracle["runtime"]["oracle"],

        candidate_pair_to_best_len={(u, v): new_len for u, v, new_len in rows},

        t_cur=t_cur,

        pivot_time_inference=int(oracle["config"].get("pivot_time", 0)),

        embeddings_dict=oracle["runtime"].get("embeddings_90"),

    )





def score_tgn(oracle: Dict[str, Any], candidate_edges, t_cur=None):

    c = core()

    return c._score_candidate_pairs_batch(

        oracle=oracle["runtime"]["oracle"],

        candidate_pair_to_best_len={

            (int(edge[0]), int(edge[1])): int(edge[3]) if len(edge) > 3 else 1 for edge in candidate_edges

        },

        t_cur=t_cur,

        pivot_time_inference=int(oracle["config"].get("pivot_time", 0)),

        embeddings_dict=None,

    )





def score_jodie(oracle: Dict[str, Any], candidate_edges, t_cur=None):

    c = core()

    return c._score_jodie_candidates_batch(

        oracle=oracle["runtime"]["oracle"],

        candidate_pair_to_best_len={

            (int(edge[0]), int(edge[1])): int(edge[3]) if len(edge) > 3 else 1 for edge in candidate_edges

        },

        t_cur=t_cur,

        topk_per_source=int(oracle["config"].get("jodie_path_topk_per_source", 10)),

    )





def run_oracle_pipeline(oracle: Dict[str, Any], data: Dict[str, Any], cfg: Any, tests_df_iter, iteration: int):

    name = oracle["code_name"]

    if name == "candidate_space":

        raise NotImplementedError("Use the RQ1 script for the controlled candidate-space oracle.")

    if name == "optimal_temporal_edge":

        raise NotImplementedError("Use the RQ2 script for the controlled optimal temporal-edge oracle.")

    if name == "n2vlp_static":

        return run_n2vlp_static(oracle, data, cfg, tests_df_iter, iteration)

    if name in {"tgn_max_time", "tgn_per_time"}:

        return run_tgn(oracle, data, cfg, tests_df_iter, iteration)

    if name in {"jodie_frozen", "jodie_update"}:

        return run_jodie(oracle, data, cfg, tests_df_iter, iteration)

    raise ValueError(f"Unknown oracle: {name}")





def run_n2vlp_static(oracle: Dict[str, Any], data: Dict[str, Any], cfg: Any, tests_df_iter, iteration: int):

    c = core()

    c.apply_config_globals(cfg, oracle_mode="static")

    return c.run_static_or_tgn_pipeline(

        data=data,

        cfg=cfg,

        method=oracle["method_label"],

        oracle_mode="static",

        oracle=oracle["runtime"]["oracle"],

        embeddings_90=oracle["runtime"]["embeddings_90"],

        lp_edge_threshold=float(oracle["runtime"]["threshold"]),

        tests_df_iter=tests_df_iter,

        iteration=iteration,

    )





def run_tgn(oracle: Dict[str, Any], data: Dict[str, Any], cfg: Any, tests_df_iter, iteration: int):

    c = core()

    cfg.tgn_edge_time_selection_mode = oracle.get("edge_time_selection_mode", "best_score_until_query_time")

    c.apply_config_globals(cfg, oracle_mode="tgn")

    c._warmup_tgn_oracle(oracle["runtime"]["oracle"], data["edges_90"])

    return c.run_static_or_tgn_pipeline(

        data=data,

        cfg=cfg,

        method=oracle["method_label"],

        oracle_mode="tgn",

        oracle=oracle["runtime"]["oracle"],

        embeddings_90=None,

        lp_edge_threshold=float(oracle["runtime"]["threshold"]),

        tests_df_iter=tests_df_iter,

        iteration=iteration,

    )





def run_jodie(oracle: Dict[str, Any], data: Dict[str, Any], cfg: Any, tests_df_iter, iteration: int):

    c = core()

    state_mode = oracle.get("state_mode", "frozen")

    c.apply_config_globals(cfg, oracle_mode="jodie_rr", jodie_state_mode=state_mode)

    c._jodie_warmup_state(oracle["runtime"]["oracle"], data["edges_90"])

    return c.run_jodie_pipeline(

        data=data,

        cfg=cfg,

        method=oracle["method_label"],

        state_mode=state_mode,

        oracle_frozen=oracle["runtime"]["oracle"],

        lp_edge_threshold=float(oracle["runtime"]["threshold"]),

        tests_df_iter=tests_df_iter,

        iteration=iteration,

    )

