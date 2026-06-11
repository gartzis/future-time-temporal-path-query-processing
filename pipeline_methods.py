from __future__ import annotations



import math

import random

import time as time_lib

import copy

from pathlib import Path

from typing import Dict, List, Tuple

from math import ceil

from collections import defaultdict

from bisect import bisect_right



import pandas as pd

import sys

import numpy as np

from sklearn.linear_model import LogisticRegression

try:

    from node2vec import Node2Vec

except ModuleNotFoundError:

    Node2Vec = None

from sklearn.metrics import roc_auc_score, average_precision_score



import pickle

import networkx as nx

import torch



BASE_DIR = Path(__file__).resolve().parent



TGN_REPO_DIR = BASE_DIR / "External" / "tgn"

sys.path.insert(0, str(TGN_REPO_DIR))



from model.tgn import TGN

from utils.data_processing import Data, compute_time_statistics

from utils.utils import RandEdgeSampler, get_neighbor_finder





sys.path.append(str(BASE_DIR / "FuturePathEstimator"))



HELPERS_DIR = BASE_DIR / "Helpers"

sys.path.insert(0, str(HELPERS_DIR))



import read_Edge_Stream as RES



from path_Prediction_Algorithms import (

    predictBiggestSPDistanceShortestPathAndDistance as PredictSP,

)





NUM_RANDOM_QUERIES = 100





NUM_DATASET_ITERATIONS = 1





QUERY_SAMPLE_SEED = 42



LP_EDGE_THRESHOLDS = [0.5]

PATH_EXIST_THRESHOLDS = [0.0]

SHORTEST_PROBA_BOUNDS = [0.0]

LK_RUNS = 20

TOP_K_RESULT_PATHS = 10





LP_ORACLE_MODE = "tgn"





RUN_TIMING_TO_MAX_FUTURE_TIMESTAMP = True





FREEZE_ANSWER_AT_ORIGINAL_FUT_TIME = True





USE_FAST_STATE_CLONE = False





FORCE_RECOMPUTE_EMBEDDINGS_PER_ITERATION = True





TRAIN_TGN_ONCE_PER_DATASET = True





BLOCK_ACCEPTED_PREDICTED_EDGES = False





USE_CACHE = True





USE_EDGE_SCORE_CACHE = USE_CACHE





EDGE_TIME_SELECTION_MODE = "best_score_until_query_time"

MAX_EDGE_TIME_CANDIDATES = None





DISABLE_MATERIALIZATION_SCORE_CACHE_IN_BEST_TIME_MODE = True





USE_EDGE_BEST_TIME_CACHE = USE_CACHE





USE_CANDIDATE_PATH_INDEX_CACHE = USE_CACHE





USE_DESTINATION_CONNECTOR_TARGETS = True

DEST_CONNECTOR_POOL_SIZE = 50

DEST_CONNECTOR_ACTIVE_FALLBACK_SIZE = 20

DEST_CONNECTOR_SOURCE_BUDGET = 5





USE_PREFIX_SUPPORTED_LANDMARKS = True



LANDMARK_CONNECTOR_POOL_SIZE = 50

LANDMARK_CONNECTOR_WEIGHT = 2.0

LANDMARK_PREFIX_PATH_WEIGHT = 1.0

LANDMARK_REACHABILITY_WEIGHT = 0.25

LANDMARK_DESTINATION_FREQUENCY_WEIGHT = 0.0

LANDMARK_REACHABILITY_MAX_DISTANCE = 3





USE_WORKLOAD_LANDMARKS = USE_PREFIX_SUPPORTED_LANDMARKS





INCLUDE_CONNECTOR_SOURCES_AS_CANDIDATE_SOURCES = True

INCLUDE_LANDMARK_CONNECTOR_SOURCES = True

INCLUDE_REACHED_LANDMARK_SOURCES = True





INCLUDE_PATH_TARGETS_IN_CONNECTOR_MODE = False





INCLUDE_REACHED_NODES_AS_COMPLETION_SOURCES = True





USE_ACTIVE_TARGET_POOL = False

ACTIVE_TARGET_POOL_SIZE = 0

USE_ACTIVE_TARGETS_ONLY_FOR_TOP_SOURCES = True

ACTIVE_TARGET_SOURCE_BUDGET = DEST_CONNECTOR_SOURCE_BUDGET





MAX_ACCEPTED_EDGES_PER_LV_PATH = None



TOPK_PRIORITY = "prefix_supported_landmark_connector_no_post_topk"





USE_SHORTCUT_GAIN_TOPK = False

USE_TARGET_SUPPORT_TOPK = False

DEMOTE_ROOT_DIRECT_EDGES = True

DISABLE_ROOT_DIRECT_EDGES = False





TESTS_DIR = Path("Data/query_tests")

TOP100_TEST_SUFFIX = ""



DATASETS = [





    ("ml_bitcoinotc_disperse_NoDuplicate_sorted.csv", -1),

]









CACHE_TAG = "cache_on" if USE_CACHE else "cache_off"



BASE_OUTPUT_DIR = Path(

    f"./results_tgn_best_time_root_direct_completion_sources_{CACHE_TAG}"

)



RESULTS_DIR = BASE_OUTPUT_DIR / f"path_prediction_results link predictor_{LP_ORACLE_MODE}"

COUNTS_DIR = BASE_OUTPUT_DIR / f"candidate_counts link predictor_{LP_ORACLE_MODE}"

EDGE_CANDIDATES_DIR = BASE_OUTPUT_DIR / f"edge_candidates_{LP_ORACLE_MODE}"

EMBEDDINGS_DIR = BASE_OUTPUT_DIR / f"embeddings_cache_{LP_ORACLE_MODE}"

PHASEA_DEBUG_DIR = BASE_OUTPUT_DIR / f"phaseA_debug_logs_{LP_ORACLE_MODE}"

TIMING_DIR = BASE_OUTPUT_DIR / f"pipeline_timing_{LP_ORACLE_MODE}"





def _iteration_query_seed(iteration: int):

    if QUERY_SAMPLE_SEED is None:

        return None

    return QUERY_SAMPLE_SEED + iteration - 1





def _normalize_embeddings(embeddings_dict):

    return {

        int(k): np.asarray(v, dtype=np.float64)

        for k, v in embeddings_dict.items()

    }





def _clone_path_record(record):

    if isinstance(record, dict):

        out = {}

        for k, v in record.items():

            if isinstance(v, list):

                out[k] = list(v)

            elif isinstance(v, dict):

                out[k] = dict(v)

            elif isinstance(v, tuple):

                out[k] = tuple(list(x) if isinstance(x, list) else dict(x) if isinstance(x, dict) else x for x in v)

            else:

                out[k] = v

        return out



    if isinstance(record, tuple):

        return tuple(

            list(x) if isinstance(x, list)

            else dict(x) if isinstance(x, dict)

            else x

            for x in record

        )



    if isinstance(record, list):

        return [

            list(x) if isinstance(x, list)

            else dict(x) if isinstance(x, dict)

            else x

            for x in record

        ]



    return record





def _clone_query_state(

    sp_pivot_base,

    L_pivot_base,

    actual_pivot_base,

    dest_paths_pivot_base,

    pivot_baseline_sp_base,

):

    sp_pivot = {k: _clone_path_record(v) for k, v in sp_pivot_base.items()}

    actual_pivot = {k: _clone_path_record(v) for k, v in actual_pivot_base.items()}

    pivot_baseline_sp = {k: _clone_path_record(v) for k, v in pivot_baseline_sp_base.items()}



    L_pivot = {

        k: [_clone_path_record(rec) for rec in v]

        for k, v in L_pivot_base.items()

    }

    dest_paths_pivot = {

        k: [_clone_path_record(rec) for rec in v]

        for k, v in dest_paths_pivot_base.items()

    }



    return sp_pivot, L_pivot, actual_pivot, dest_paths_pivot, pivot_baseline_sp





def _fresh_query_state(

    sp_pivot_base,

    L_pivot_base,

    actual_pivot_base,

    dest_paths_pivot_base,

    pivot_baseline_sp_base,

):

    if USE_FAST_STATE_CLONE:

        return _clone_query_state(

            sp_pivot_base,

            L_pivot_base,

            actual_pivot_base,

            dest_paths_pivot_base,

            pivot_baseline_sp_base,

        )



    return (

        copy.deepcopy(sp_pivot_base),

        copy.deepcopy(L_pivot_base),

        copy.deepcopy(actual_pivot_base),

        copy.deepcopy(dest_paths_pivot_base),

        copy.deepcopy(pivot_baseline_sp_base),

    )





def _first_future_appearance_times(edges_future):

    first_time = {}

    for (u, v, t, _) in edges_future:

        key = (u, v)

        if key not in first_time or t < first_time[key]:

            first_time[key] = t

    return first_time





def _time_feature_vector(t, pivot_time, time_scale):

    if t is None or pivot_time is None:

        return np.zeros(2, dtype=float)



    scale = max(1.0, float(time_scale))

    delta = max(0.0, float(t - pivot_time))

    z = delta / scale



    return np.asarray([

        np.log1p(z),

        z / (1.0 + z),

    ], dtype=float)





def _hadamard_edge_embedding(u, v, embeddings_dict):

    emb_u = embeddings_dict.get(u)

    emb_v = embeddings_dict.get(v)



    if emb_u is None or emb_v is None:

        return None



    return emb_u * emb_v





def _edge_feature_vector(u, v, embeddings_dict, mode="static",

                         t=None, pivot_time=None, time_scale=None):

    base = _hadamard_edge_embedding(u, v, embeddings_dict)

    if base is None:

        return None



    if mode == "static":

        return base



    if mode == "time_features":

        tf = _time_feature_vector(t, pivot_time, time_scale)

        return np.concatenate([base, tf])



    raise ValueError(f"Unknown LP oracle mode: {mode}")





def _sample_negative_edge_times_before_first_appearance(

    vertex_list,

    forbidden_edges,

    future_times,

    first_future_time,

    num_samples,

    rng,

):

    future_times = sorted(future_times)

    time_to_idx = {t: i for i, t in enumerate(future_times)}



    negatives = []

    used = set()



    while len(negatives) < num_samples:

        u = rng.choice(vertex_list)

        v = rng.choice(vertex_list)



        if u == v:

            continue

        if (u, v) in forbidden_edges:

            continue



        t_first = first_future_time.get((u, v), None)



        if t_first is None:

            max_idx = len(future_times) - 1

        else:

            max_idx = time_to_idx[t_first] - 1



        if max_idx < 0:

            continue



        t = future_times[rng.randint(0, max_idx)]



        key = (u, v, t)

        if key in used:

            continue



        used.add(key)

        negatives.append((u, v, t))



    return negatives





def _edge_set_without_time(edge_list):

    return {(u, v) for (u, v, _, _) in edge_list}





def _sample_negative_edges(vertex_list, forbidden_edges, num_samples, rng):

    negatives = []

    used = set(forbidden_edges)



    while len(negatives) < num_samples:

        u = rng.choice(vertex_list)

        v = rng.choice(vertex_list)

        if u == v:

            continue

        if (u, v) in used:

            continue

        used.add((u, v))

        negatives.append((u, v))



    return negatives





def _train_lp_logreg_hadamard(edges_train, edges_future, vertex_list, embeddings_dict, seed=42):

    rng = random.Random(seed)



    train_edge_set = _edge_set_without_time(edges_train)

    positive_edges = list({

        (u, v)

        for (u, v, _, _) in edges_future

        if u in embeddings_dict and v in embeddings_dict

    })

    negative_edges = _sample_negative_edges(

        vertex_list,

        forbidden_edges=train_edge_set.union(set(positive_edges)),

        num_samples=len(positive_edges),

        rng=rng,

    )



    X = []

    y = []



    for (u, v) in positive_edges:

        feat = _hadamard_edge_embedding(u, v, embeddings_dict)

        if feat is not None:

            X.append(feat)

            y.append(1)



    for (u, v) in negative_edges:

        feat = _hadamard_edge_embedding(u, v, embeddings_dict)

        if feat is not None:

            X.append(feat)

            y.append(0)



    if len(X) == 0:

        raise ValueError("LP training set is empty after embedding filtering.")



    if len(set(y)) < 2:

        raise ValueError("LP training set has fewer than 2 classes after embedding filtering.")



    clf = LogisticRegression(max_iter=2000, random_state=seed)

    clf.fit(np.asarray(X), np.asarray(y))

    return clf





def _train_lp_oracle(

    mode,

    edges_train,

    edges_future,

    vertex_list,

    embeddings_dict,

    pivot_time,

    future_times_train,

    seed=42,

):

    time_scale = 1.0

    if future_times_train:

        time_scale = max(1.0, float(max(future_times_train) - pivot_time))



    if mode == "static":

        model = _train_lp_logreg_hadamard(

            edges_train=edges_train,

            edges_future=edges_future,

            vertex_list=vertex_list,

            embeddings_dict=embeddings_dict,

            seed=seed,

        )

        return {

            "mode": mode,

            "model": model,

            "embeddings": embeddings_dict,

            "pivot_time": pivot_time,

            "future_times_train": list(sorted(future_times_train)),

            "time_scale": time_scale,

        }



    if mode == "time_features":

        rng = random.Random(seed)



        train_edge_set = _edge_set_without_time(edges_train)

        first_future_time = _first_future_appearance_times(edges_future)



        positive_examples = [

            (u, v, t_first)

            for ((u, v), t_first) in first_future_time.items()

            if u in embeddings_dict and v in embeddings_dict

        ]



        positive_pairs = {(u, v) for (u, v, _) in positive_examples}



        negative_examples = _sample_negative_edge_times_before_first_appearance(

            vertex_list=vertex_list,

            forbidden_edges=train_edge_set.union(positive_pairs),

            future_times=future_times_train,

            first_future_time=first_future_time,

            num_samples=len(positive_examples),

            rng=rng,

        )



        X = []

        y = []



        for (u, v, t) in positive_examples:

            feat = _edge_feature_vector(

                u, v, embeddings_dict,

                mode=mode,

                t=t,

                pivot_time=pivot_time,

                time_scale=time_scale,

            )

            if feat is not None:

                X.append(feat)

                y.append(1)



        for (u, v, t) in negative_examples:

            feat = _edge_feature_vector(

                u, v, embeddings_dict,

                mode=mode,

                t=t,

                pivot_time=pivot_time,

                time_scale=time_scale,

            )

            if feat is not None:

                X.append(feat)

                y.append(0)



        if len(X) == 0:

            raise ValueError("LP training set is empty after embedding filtering.")

        if len(set(y)) < 2:

            raise ValueError("LP training set has fewer than 2 classes after filtering.")



        model = LogisticRegression(max_iter=2000, random_state=seed)

        model.fit(np.asarray(X), np.asarray(y))



        return {

            "mode": mode,

            "model": model,

            "embeddings": embeddings_dict,

            "pivot_time": pivot_time,

            "future_times_train": list(sorted(future_times_train)),

            "time_scale": time_scale,

        }



    raise ValueError(f"Unknown LP oracle mode: {mode}")





def _score_lp_oracle(

    oracle,

    u,

    v,

    t_cur,

    pivot_time_inference,

    future_times_inference,

    embeddings_dict=None,

):

    use_embeddings = oracle["embeddings"] if embeddings_dict is None else embeddings_dict



    feat = _edge_feature_vector(

        u, v,

        use_embeddings,

        mode=oracle["mode"],

        t=t_cur,

        pivot_time=pivot_time_inference,

        time_scale=oracle.get("time_scale", 1.0),

    )

    if feat is None:

        return None



    return float(oracle["model"].predict_proba([feat])[0][1])





def _score_candidate_pairs_batch(

    oracle,

    candidate_pair_to_best_len,

    t_cur,

    pivot_time_inference,

    embeddings_dict=None,

):

    if oracle["mode"] == "tgn":

        pairs = [

            (u, v, new_len)

            for (u, v), new_len in candidate_pair_to_best_len.items()

        ]



        return _score_tgn_pairs_no_update(

            oracle=oracle,

            pairs=pairs,

            t_cur=t_cur,

        )





    use_embeddings = oracle["embeddings"] if embeddings_dict is None else embeddings_dict

    mode = oracle["mode"]

    time_scale = oracle.get("time_scale", 1.0)



    kept = []

    X = []



    if mode == "time_features":

        tf = _time_feature_vector(t_cur, pivot_time_inference, time_scale)

    else:

        tf = None



    for (u, v), new_len in candidate_pair_to_best_len.items():

        emb_u = use_embeddings.get(u)

        emb_v = use_embeddings.get(v)



        if emb_u is None or emb_v is None:

            continue



        base = emb_u * emb_v



        if mode == "static":

            feat = base

        elif mode == "time_features":

            feat = np.concatenate([base, tf])

        else:

            raise ValueError(f"Unknown LP oracle mode: {mode}")



        kept.append((u, v, new_len))

        X.append(feat)



    if not X:

        return []



    probs = oracle["model"].predict_proba(np.asarray(X))[:, 1]



    return [

        (u, v, new_len, float(p_hat))

        for (u, v, new_len), p_hat in zip(kept, probs)

    ]





def _candidate_time_grid_until_query_time(

    future_times,

    t_start,

    query_future_time,

    max_edge_time_candidates=None,

):

    t_start = int(t_start)

    query_future_time = int(query_future_time)



    times = [

        int(t)

        for t in future_times

        if int(t) >= t_start and int(t) <= query_future_time

    ]



    if not times and t_start <= query_future_time:

        times = [t_start]



    if max_edge_time_candidates is not None:

        k = int(max_edge_time_candidates)

        if k > 0 and len(times) > k:





            times = times[:k]



    return times





def _edge_best_time_cache_key(oracle_mode, u, v, candidate_times):

    return (

        str(oracle_mode),

        int(u),

        int(v),

        tuple(int(t) for t in candidate_times),

    )





def _score_candidate_pairs_best_time_until_query_cached(

    oracle,

    candidate_pair_to_best_len,

    candidate_times,

    pivot_time_inference,

    embeddings_dict,

    edge_score_cache,

    edge_cache_low_by_tu,

    edge_cache_high_by_tu,

    lp_edge_threshold,

    oracle_mode,

    edge_best_time_cache=None,

    use_edge_score_cache=True,

    use_edge_best_time_cache=True,

):

    candidate_times = [int(t) for t in candidate_times]



    if not candidate_pair_to_best_len or not candidate_times:

        return [], {}, {

            "edge_time_num_candidate_times": len(candidate_times),

            "edge_time_pair_time_cache_hits": 0,

            "edge_time_pair_time_cache_misses": 0,

            "edge_time_pair_time_scored": 0,

            "edge_time_best_cache_hits": 0,

            "edge_time_best_cache_misses": 0,

            "edge_time_best_cache_stores": 0,

            "edge_time_best_cache_size": len(edge_best_time_cache) if edge_best_time_cache is not None else 0,

            "edge_time_best_pairs": 0,

            "edge_time_min_selected_time": -1,

            "edge_time_max_selected_time": -1,

        }



    best_score_by_pair = {}

    best_time_by_pair = {}



    pair_time_cache_hits = 0

    pair_time_cache_misses = 0

    pair_time_scored = 0



    best_cache_hits = 0

    best_cache_misses = 0

    best_cache_stores = 0



    if edge_best_time_cache is None:

        edge_best_time_cache = {}



    def _maybe_update_best(u, v, t_eval, p_hat):

        pair_key = (int(u), int(v))

        p_hat = float(p_hat)

        t_eval = int(t_eval)

        old_score = best_score_by_pair.get(pair_key)

        old_time = best_time_by_pair.get(pair_key)





        if (

            old_score is None

            or p_hat > old_score + 1e-12

            or (abs(p_hat - old_score) <= 1e-12 and (old_time is None or t_eval < old_time))

        ):

            best_score_by_pair[pair_key] = p_hat

            best_time_by_pair[pair_key] = t_eval



    pairs_to_scan = {}

    for (u, v), new_len in candidate_pair_to_best_len.items():

        u = int(u)

        v = int(v)



        if use_edge_best_time_cache:

            best_cache_key = _edge_best_time_cache_key(oracle_mode, u, v, candidate_times)

            cached_best = edge_best_time_cache.get(best_cache_key)

            if cached_best is not None:

                best_cache_hits += 1

                best_t, best_p = cached_best

                _maybe_update_best(u, v, int(best_t), float(best_p))

                continue



            best_cache_misses += 1



        pairs_to_scan[(u, v)] = int(new_len)



    for t_eval in candidate_times:

        missing_pair_to_len = {}



        for (u, v), new_len in pairs_to_scan.items():

            u = int(u)

            v = int(v)

            cache_key = _edge_score_cache_key(oracle_mode, u, v, t_eval)



            if use_edge_score_cache and cache_key in edge_score_cache:

                pair_time_cache_hits += 1

                _maybe_update_best(u, v, t_eval, edge_score_cache[cache_key])

            else:

                pair_time_cache_misses += 1

                missing_pair_to_len[(u, v)] = int(new_len)



        if not missing_pair_to_len:

            continue



        scored_at_time = _score_candidate_pairs_batch(

            oracle=oracle,

            candidate_pair_to_best_len=missing_pair_to_len,

            t_cur=t_eval,

            pivot_time_inference=pivot_time_inference,

            embeddings_dict=embeddings_dict,

        )

        pair_time_scored += len(scored_at_time)



        for u, v, _new_len, p_hat in scored_at_time:

            u = int(u)

            v = int(v)

            p_hat = float(p_hat)



            if use_edge_score_cache:

                _store_edge_score_in_indexes(

                    edge_score_cache=edge_score_cache,

                    edge_cache_low_by_tu=edge_cache_low_by_tu,

                    edge_cache_high_by_tu=edge_cache_high_by_tu,

                    oracle_mode=oracle_mode,

                    u=u,

                    v=v,

                    t_cur=t_eval,

                    p_hat=p_hat,

                    lp_edge_threshold=lp_edge_threshold,

                )



            _maybe_update_best(u, v, t_eval, p_hat)



    if use_edge_best_time_cache:

        for (u, v), _new_len in pairs_to_scan.items():

            pair_key = (int(u), int(v))

            if pair_key not in best_score_by_pair or pair_key not in best_time_by_pair:

                continue

            best_cache_key = _edge_best_time_cache_key(oracle_mode, u, v, candidate_times)

            if best_cache_key not in edge_best_time_cache:

                edge_best_time_cache[best_cache_key] = (

                    int(best_time_by_pair[pair_key]),

                    float(best_score_by_pair[pair_key]),

                )

                best_cache_stores += 1



    best_scored_candidates = []

    for pair_key, p_hat in best_score_by_pair.items():

        u, v = pair_key

        new_len = candidate_pair_to_best_len.get(pair_key)

        if new_len is None:

            continue

        best_scored_candidates.append((int(u), int(v), int(new_len), float(p_hat)))



    selected_times = list(best_time_by_pair.values())

    stats = {

        "edge_time_num_candidate_times": len(candidate_times),

        "edge_time_pair_time_cache_hits": pair_time_cache_hits,

        "edge_time_pair_time_cache_misses": pair_time_cache_misses,

        "edge_time_pair_time_scored": pair_time_scored,

        "edge_time_best_cache_hits": best_cache_hits,

        "edge_time_best_cache_misses": best_cache_misses,

        "edge_time_best_cache_stores": best_cache_stores,

        "edge_time_best_cache_size": len(edge_best_time_cache),

        "edge_time_best_pairs": len(best_scored_candidates),

        "edge_time_min_selected_time": min(selected_times) if selected_times else -1,

        "edge_time_max_selected_time": max(selected_times) if selected_times else -1,

    }



    return best_scored_candidates, best_time_by_pair, stats





def _evaluate_lp_oracle_on_window(

    oracle,

    edges_prefix,

    edges_future,

    vertex_list,

    embeddings_dict,

    pivot_time,

    future_times_eval,

    lp_edge_thresholds,

    seed=42,

):

    rng = random.Random(seed)

    future_times_eval = sorted(future_times_eval)



    examples = []



    if oracle["mode"] == "static":

        prefix_edge_set = _edge_set_without_time(edges_prefix)



        positive_edges = list({

            (u, v)

            for (u, v, _, _) in edges_future

            if u in embeddings_dict and v in embeddings_dict

        })



        negative_edges = _sample_negative_edges(

            vertex_list=vertex_list,

            forbidden_edges=prefix_edge_set.union(set(positive_edges)),

            num_samples=len(positive_edges),

            rng=rng,

        )



        for (u, v) in positive_edges:

            score = _score_lp_oracle(

                oracle=oracle,

                u=u,

                v=v,

                t_cur=None,

                pivot_time_inference=pivot_time,

                future_times_inference=future_times_eval,

                embeddings_dict=embeddings_dict,

            )

            if score is not None:

                examples.append((1, score))



        for (u, v) in negative_edges:

            score = _score_lp_oracle(

                oracle=oracle,

                u=u,

                v=v,

                t_cur=None,

                pivot_time_inference=pivot_time,

                future_times_inference=future_times_eval,

                embeddings_dict=embeddings_dict,

            )

            if score is not None:

                examples.append((0, score))



    elif oracle["mode"] == "time_features":

        prefix_edge_set = _edge_set_without_time(edges_prefix)

        first_future_time = _first_future_appearance_times(edges_future)



        positive_examples = [

            (u, v, t_first)

            for ((u, v), t_first) in first_future_time.items()

            if u in embeddings_dict and v in embeddings_dict

        ]



        positive_pairs = {(u, v) for (u, v, _) in positive_examples}



        negative_examples = _sample_negative_edge_times_before_first_appearance(

            vertex_list=vertex_list,

            forbidden_edges=prefix_edge_set.union(positive_pairs),

            future_times=future_times_eval,

            first_future_time=first_future_time,

            num_samples=len(positive_examples),

            rng=rng,

        )



        for (u, v, t) in positive_examples:

            score = _score_lp_oracle(

                oracle=oracle,

                u=u,

                v=v,

                t_cur=t,

                pivot_time_inference=pivot_time,

                future_times_inference=future_times_eval,

                embeddings_dict=embeddings_dict,

            )

            if score is not None:

                examples.append((1, score))



        for (u, v, t) in negative_examples:

            score = _score_lp_oracle(

                oracle=oracle,

                u=u,

                v=v,

                t_cur=t,

                pivot_time_inference=pivot_time,

                future_times_inference=future_times_eval,

                embeddings_dict=embeddings_dict,

            )

            if score is not None:

                examples.append((0, score))



    elif oracle["mode"] == "tgn":

        prefix_edge_set = _edge_set_without_time(edges_prefix)

        first_future_time = _first_future_appearance_times(edges_future)



        positive_examples = [

            (u, v, t_first)

            for ((u, v), t_first) in first_future_time.items()

            if u in oracle["node_to_tgn"] and v in oracle["node_to_tgn"]

        ]



        positive_pairs = {(u, v) for (u, v, _) in positive_examples}



        negative_examples = _sample_negative_edge_times_before_first_appearance(

            vertex_list=vertex_list,

            forbidden_edges=prefix_edge_set.union(positive_pairs),

            future_times=future_times_eval,

            first_future_time=first_future_time,

            num_samples=len(positive_examples),

            rng=rng,

        )





        by_time = {}



        for (u, v, t) in positive_examples:

            by_time.setdefault(t, []).append((1, u, v))



        for (u, v, t) in negative_examples:

            if u in oracle["node_to_tgn"] and v in oracle["node_to_tgn"]:

                by_time.setdefault(t, []).append((0, u, v))



        for t, items in by_time.items():

            candidate_pair_to_best_len = {

                (u, v): 1

                for (_, u, v) in items

            }



            scored = _score_candidate_pairs_batch(

                oracle=oracle,

                candidate_pair_to_best_len=candidate_pair_to_best_len,

                t_cur=t,

                pivot_time_inference=pivot_time,

                embeddings_dict=None,

            )



            score_dict = {

                (u, v): p_hat

                for (u, v, _, p_hat) in scored

            }



            for y, u, v in items:

                s = score_dict.get((u, v))

                if s is not None:

                    examples.append((y, s))

    else:

        raise ValueError(f"Unknown LP oracle mode: {oracle['mode']}")



    if not examples:

        raise ValueError("Validation/test example set is empty.")



    y_true = [y for (y, _) in examples]

    y_score = [s for (_, s) in examples]



    if len(set(y_true)) < 2:

        auc = np.nan

    else:

        auc = float(roc_auc_score(y_true, y_score))



    rows = []

    best_threshold = lp_edge_thresholds[0]

    best_sum = -1.0



    for thr in lp_edge_thresholds:

        tp = sum(1 for y, s in examples if y == 1 and s >= thr)

        fp = sum(1 for y, s in examples if y == 0 and s >= thr)

        fn = sum(1 for y, s in examples if y == 1 and s < thr)

        tn = sum(1 for y, s in examples if y == 0 and s < thr)



        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        acc = (tp + tn) / max(1, tp + tn + fp + fn)



        rows.append({

            "threshold": thr,

            "tp": tp,

            "fp": fp,

            "fn": fn,

            "tn": tn,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "accuracy": acc,

            "auc": auc,

        })



        new_sum = precision + recall + acc

        if new_sum > best_sum:

            best_threshold = thr

            best_sum = new_sum



    return best_threshold, pd.DataFrame(rows)





TGN_NODE_DIM = 128

TGN_EDGE_DIM = 1

TGN_BATCH_SIZE = 200

TGN_EPOCHS = 50

TGN_USE_EARLY_STOPPING = False

TGN_LR = 1e-4

TGN_NUM_NEIGHBORS = 10

TGN_NUM_LAYERS = 1

TGN_NUM_HEADS = 2

TGN_DROPOUT = 0.1

TGN_BACKPROP_EVERY = 1

TGN_PATIENCE = 5





def _build_tgn_node_mapping(*edge_lists):

    nodes = sorted({

        int(x)

        for edge_list in edge_lists

        for (u, v, _, _) in edge_list

        for x in (u, v)

    })



    node_to_tgn = {node: i + 1 for i, node in enumerate(nodes)}

    tgn_to_node = {i + 1: node for i, node in enumerate(nodes)}



    return node_to_tgn, tgn_to_node





def _make_tgn_data(edge_list, node_to_tgn):

    rows = []



    for u, v, t, _ in edge_list:

        u = int(u)

        v = int(v)

        t = float(t)



        if u not in node_to_tgn or v not in node_to_tgn:

            continue



        rows.append((node_to_tgn[u], node_to_tgn[v], t))



    rows.sort(key=lambda x: x[2])



    if not rows:

        return Data(

            sources=np.asarray([], dtype=np.int64),

            destinations=np.asarray([], dtype=np.int64),

            timestamps=np.asarray([], dtype=np.float32),

            edge_idxs=np.asarray([], dtype=np.int64),

            labels=np.asarray([], dtype=np.float32),

        )



    sources = np.asarray([r[0] for r in rows], dtype=np.int64)

    destinations = np.asarray([r[1] for r in rows], dtype=np.int64)

    timestamps = np.asarray([r[2] for r in rows], dtype=np.float32)



    edge_idxs = np.ones(len(rows), dtype=np.int64)

    labels = np.zeros(len(rows), dtype=np.float32)



    return Data(sources, destinations, timestamps, edge_idxs, labels)





def _make_tgn_features(n_tgn_nodes, node_dim=TGN_NODE_DIM, edge_dim=TGN_EDGE_DIM):

    node_features = np.zeros((n_tgn_nodes + 1, node_dim), dtype=np.float32)



    edge_features = np.zeros((2, edge_dim), dtype=np.float32)

    edge_features[1, 0] = 1.0



    return node_features, edge_features





def _safe_time_stats(data):

    mean_src, std_src, mean_dst, std_dst = compute_time_statistics(

        data.sources,

        data.destinations,

        data.timestamps,

    )



    std_src = float(std_src) if std_src > 1e-12 else 1.0

    std_dst = float(std_dst) if std_dst > 1e-12 else 1.0



    return float(mean_src), std_src, float(mean_dst), std_dst





def _threshold_table_from_scores(y_true, y_score, lp_edge_thresholds):

    if len(set(y_true)) < 2:

        auc = np.nan

        ap = np.nan

    else:

        auc = float(roc_auc_score(y_true, y_score))

        ap = float(average_precision_score(y_true, y_score))



    rows = []

    best_threshold = lp_edge_thresholds[0]

    best_sum = -1.0



    for thr in lp_edge_thresholds:

        tp = sum(1 for y, s in zip(y_true, y_score) if y == 1 and s >= thr)

        fp = sum(1 for y, s in zip(y_true, y_score) if y == 0 and s >= thr)

        fn = sum(1 for y, s in zip(y_true, y_score) if y == 1 and s < thr)

        tn = sum(1 for y, s in zip(y_true, y_score) if y == 0 and s < thr)



        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        acc = (tp + tn) / max(1, tp + tn + fp + fn)



        rows.append({

            "threshold": thr,

            "tp": tp,

            "fp": fp,

            "fn": fn,

            "tn": tn,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "accuracy": acc,

            "auc": auc,

            "ap": ap,

        })



        score = precision + recall + acc

        if score > best_sum:

            best_sum = score

            best_threshold = thr



    return best_threshold, pd.DataFrame(rows)





def _make_tgn_oracle_dict(

    tgn,

    node_to_tgn,

    tgn_to_node,

    n_tgn_nodes,

    device,

    batch_size,

):

    return {

        "mode": "tgn",

        "model": tgn,

        "node_to_tgn": node_to_tgn,

        "tgn_to_node": tgn_to_node,

        "n_tgn_nodes": n_tgn_nodes,

        "device": device,

        "batch_size": batch_size,

    }





def _evaluate_tgn_edge_scores_official_update(

    oracle,

    edges_future,

    negative_source_edges,

    neighbor_source_edges=None,

    seed=42,

):

    tgn = oracle["model"]

    node_to_tgn = oracle["node_to_tgn"]

    n_tgn_nodes = oracle["n_tgn_nodes"]

    batch_size = oracle["batch_size"]



    eval_data = _make_tgn_data(edges_future, node_to_tgn)

    if eval_data.n_interactions == 0:

        raise ValueError("TGN evaluation data is empty.")



    if neighbor_source_edges is not None:

        ngh_data = _make_tgn_data(neighbor_source_edges, node_to_tgn)

        ngh_finder = get_neighbor_finder(

            ngh_data,

            uniform=False,

            max_node_idx=n_tgn_nodes,

        )

        tgn.set_neighbor_finder(ngh_finder)



    neg_data = _make_tgn_data(negative_source_edges, node_to_tgn)

    if neg_data.n_interactions == 0:

        neg_data = eval_data



    neg_sampler = RandEdgeSampler(

        neg_data.sources,

        neg_data.destinations,

        seed=seed,

    )



    tgn.eval()



    y_true = []

    y_score = []



    num_instances = eval_data.n_interactions

    num_batches = math.ceil(num_instances / batch_size)



    with torch.no_grad():

        for batch_idx in range(num_batches):

            start_idx = batch_idx * batch_size

            end_idx = min(num_instances, start_idx + batch_size)



            sources_batch = eval_data.sources[start_idx:end_idx]

            destinations_batch = eval_data.destinations[start_idx:end_idx]

            timestamps_batch = eval_data.timestamps[start_idx:end_idx]

            edge_idxs_batch = eval_data.edge_idxs[start_idx:end_idx]



            original_size = len(sources_batch)

            if original_size == 0:

                continue



            _, negatives_batch = neg_sampler.sample(original_size)



            if original_size == 1:

                sources_batch = np.concatenate([sources_batch, sources_batch])

                destinations_batch = np.concatenate([destinations_batch, destinations_batch])

                timestamps_batch = np.concatenate([timestamps_batch, timestamps_batch])

                edge_idxs_batch = np.concatenate([edge_idxs_batch, edge_idxs_batch])

                negatives_batch = np.concatenate([negatives_batch, negatives_batch])



            pos_prob, neg_prob = tgn.compute_edge_probabilities(

                sources_batch,

                destinations_batch,

                negatives_batch,

                timestamps_batch,

                edge_idxs_batch,

                TGN_NUM_NEIGHBORS,

            )



            pos_scores = pos_prob.detach().cpu().numpy().reshape(-1)[:original_size]

            neg_scores = neg_prob.detach().cpu().numpy().reshape(-1)[:original_size]



            y_true.extend([1] * original_size)

            y_score.extend([float(x) for x in pos_scores])



            y_true.extend([0] * original_size)

            y_score.extend([float(x) for x in neg_scores])



    if not y_true:

        raise ValueError("TGN evaluation produced no examples.")



    return y_true, y_score





def _evaluate_tgn_oracle_on_window_official_update(

    oracle,

    edges_future,

    negative_source_edges,

    lp_edge_thresholds,

    neighbor_source_edges=None,

    seed=42,

):

    y_true, y_score = _evaluate_tgn_edge_scores_official_update(

        oracle=oracle,

        edges_future=edges_future,

        negative_source_edges=negative_source_edges,

        neighbor_source_edges=neighbor_source_edges,

        seed=seed,

    )



    return _threshold_table_from_scores(

        y_true=y_true,

        y_score=y_score,

        lp_edge_thresholds=lp_edge_thresholds,

    )





def _train_tgn_oracle(

    edges_train,

    edges_prefix_for_nodes,

    edges_val=None,

    seed=42,

    device=None,

    n_epochs=TGN_EPOCHS,

    batch_size=TGN_BATCH_SIZE,

):

    if seed is None:

        seed = random.randint(0, 2**31 - 1)



    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)



    if device is None:

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    node_to_tgn, tgn_to_node = _build_tgn_node_mapping(edges_prefix_for_nodes)

    n_tgn_nodes = max(node_to_tgn.values()) if node_to_tgn else 0



    train_data = _make_tgn_data(edges_train, node_to_tgn)

    full_prefix_data = _make_tgn_data(edges_prefix_for_nodes, node_to_tgn)



    if train_data.n_interactions == 0:

        raise ValueError("TGN training data is empty.")



    node_features, edge_features = _make_tgn_features(n_tgn_nodes)



    train_ngh_finder = get_neighbor_finder(

        train_data,

        uniform=False,

        max_node_idx=n_tgn_nodes,

    )



    full_prefix_ngh_finder = get_neighbor_finder(

        full_prefix_data,

        uniform=False,

        max_node_idx=n_tgn_nodes,

    )





    mean_src, std_src, mean_dst, std_dst = _safe_time_stats(full_prefix_data)



    tgn = TGN(

        neighbor_finder=train_ngh_finder,

        node_features=node_features,

        edge_features=edge_features,

        device=device,

        n_layers=TGN_NUM_LAYERS,

        n_heads=TGN_NUM_HEADS,

        dropout=TGN_DROPOUT,

        use_memory=True,

        memory_update_at_start=True,

        message_dimension=100,

        memory_dimension=TGN_NODE_DIM,

        embedding_module_type="graph_attention",

        message_function="identity",

        aggregator_type="last",

        memory_updater_type="gru",

        n_neighbors=TGN_NUM_NEIGHBORS,

        mean_time_shift_src=mean_src,

        std_time_shift_src=std_src,

        mean_time_shift_dst=mean_dst,

        std_time_shift_dst=std_dst,

    ).to(device)



    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(tgn.parameters(), lr=TGN_LR)

    train_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)



    num_instances = train_data.n_interactions

    num_batches = math.ceil(num_instances / batch_size)



    best_state_dict = None

    best_val_ap = -1.0

    bad_rounds = 0



    oracle_tmp = _make_tgn_oracle_dict(

        tgn=tgn,

        node_to_tgn=node_to_tgn,

        tgn_to_node=tgn_to_node,

        n_tgn_nodes=n_tgn_nodes,

        device=device,

        batch_size=batch_size,

    )



    for epoch in range(n_epochs):

        tgn.train()

        tgn.set_neighbor_finder(train_ngh_finder)



        if tgn.use_memory:

            tgn.memory.__init_memory__()



        epoch_losses = []



        for k in range(0, num_batches, TGN_BACKPROP_EVERY):

            loss = 0.0

            used_batches = 0



            optimizer.zero_grad()



            for j in range(TGN_BACKPROP_EVERY):

                batch_idx = k + j

                if batch_idx >= num_batches:

                    continue



                start_idx = batch_idx * batch_size

                end_idx = min(num_instances, start_idx + batch_size)



                sources_batch = train_data.sources[start_idx:end_idx]

                destinations_batch = train_data.destinations[start_idx:end_idx]

                timestamps_batch = train_data.timestamps[start_idx:end_idx]

                edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]



                size = len(sources_batch)

                if size < 2:

                    continue



                _, negatives_batch = train_sampler.sample(size)



                pos_label = torch.ones(size, dtype=torch.float32, device=device)

                neg_label = torch.zeros(size, dtype=torch.float32, device=device)



                tgn.train()



                pos_prob, neg_prob = tgn.compute_edge_probabilities(

                    sources_batch,

                    destinations_batch,

                    negatives_batch,

                    timestamps_batch,

                    edge_idxs_batch,

                    TGN_NUM_NEIGHBORS,

                )



                batch_loss = (

                    criterion(pos_prob.squeeze(), pos_label)

                    + criterion(neg_prob.squeeze(), neg_label)

                )



                loss = loss + batch_loss

                used_batches += 1



            if used_batches == 0:

                continue



            loss = loss / used_batches

            loss.backward()

            optimizer.step()



            if tgn.use_memory:

                tgn.memory.detach_memory()



            epoch_losses.append(float(loss.item()))



        mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan



        if edges_val is not None and len(edges_val) > 0:

            tgn.eval()

            tgn.set_neighbor_finder(full_prefix_ngh_finder)



            train_memory_backup = None

            if tgn.use_memory:

                train_memory_backup = tgn.memory.backup_memory()



            y_true, y_score = _evaluate_tgn_edge_scores_official_update(

                oracle=oracle_tmp,

                edges_future=edges_val,

                negative_source_edges=edges_prefix_for_nodes,

                neighbor_source_edges=edges_prefix_for_nodes,

                seed=seed,

            )



            val_ap = (

                float(average_precision_score(y_true, y_score))

                if len(set(y_true)) >= 2

                else np.nan

            )

            val_auc = (

                float(roc_auc_score(y_true, y_score))

                if len(set(y_true)) >= 2

                else np.nan

            )



            if tgn.use_memory and train_memory_backup is not None:

                tgn.memory.restore_memory(train_memory_backup)



            print(

                f"[TGN] epoch={epoch + 1}/{n_epochs} "

                f"loss={mean_train_loss:.6f} val_ap={val_ap:.6f} val_auc={val_auc:.6f}"

            )



            monitor_value = val_ap if not np.isnan(val_ap) else -1.0





            best_state_dict = copy.deepcopy(tgn.state_dict())



            if TGN_USE_EARLY_STOPPING:

                monitor_value = val_ap if not np.isnan(val_ap) else -1.0



                if monitor_value > best_val_ap:

                    best_val_ap = monitor_value

                    best_state_dict = copy.deepcopy(tgn.state_dict())

                    bad_rounds = 0

                else:

                    bad_rounds += 1



                if bad_rounds >= TGN_PATIENCE:

                    print(

                        f"[TGN] early stopping after epoch {epoch + 1}; "

                        f"best_val_ap={best_val_ap:.6f}"

                    )

                    break



        else:

            print(f"[TGN] epoch={epoch + 1}/{n_epochs} loss={mean_train_loss:.6f}")

            best_state_dict = copy.deepcopy(tgn.state_dict())



    if best_state_dict is not None:

        tgn.load_state_dict(best_state_dict)



    tgn.eval()

    tgn.set_neighbor_finder(full_prefix_ngh_finder)



    return _make_tgn_oracle_dict(

        tgn=tgn,

        node_to_tgn=node_to_tgn,

        tgn_to_node=tgn_to_node,

        n_tgn_nodes=n_tgn_nodes,

        device=device,

        batch_size=batch_size,

    )





def _warmup_tgn_oracle(oracle, edges_prefix):

    tgn = oracle["model"]

    node_to_tgn = oracle["node_to_tgn"]

    n_tgn_nodes = oracle["n_tgn_nodes"]

    batch_size = oracle["batch_size"]



    prefix_data = _make_tgn_data(edges_prefix, node_to_tgn)



    prefix_ngh_finder = get_neighbor_finder(

        prefix_data,

        uniform=False,

        max_node_idx=n_tgn_nodes,

    )



    tgn.eval()

    tgn.set_neighbor_finder(prefix_ngh_finder)



    if tgn.use_memory:

        tgn.memory.__init_memory__()



    num_instances = prefix_data.n_interactions

    num_batches = math.ceil(num_instances / batch_size)



    with torch.no_grad():

        for batch_idx in range(num_batches):

            start_idx = batch_idx * batch_size

            end_idx = min(num_instances, start_idx + batch_size)



            sources_batch = prefix_data.sources[start_idx:end_idx]

            destinations_batch = prefix_data.destinations[start_idx:end_idx]

            timestamps_batch = prefix_data.timestamps[start_idx:end_idx]

            edge_idxs_batch = prefix_data.edge_idxs[start_idx:end_idx]



            if len(sources_batch) == 0:

                continue



            if len(sources_batch) == 1:

                sources_batch = np.concatenate([sources_batch, sources_batch])

                destinations_batch = np.concatenate([destinations_batch, destinations_batch])

                timestamps_batch = np.concatenate([timestamps_batch, timestamps_batch])

                edge_idxs_batch = np.concatenate([edge_idxs_batch, edge_idxs_batch])



            dummy_negatives = destinations_batch.copy()



            tgn.compute_temporal_embeddings(

                sources_batch,

                destinations_batch,

                dummy_negatives,

                timestamps_batch,

                edge_idxs_batch,

                TGN_NUM_NEIGHBORS,

            )



        if tgn.use_memory:

            all_nodes = np.arange(tgn.n_nodes)

            tgn.update_memory(all_nodes, tgn.memory.messages)

            tgn.memory.clear_messages(all_nodes)





def _compute_tgn_embeddings_no_update(oracle, nodes, times, role):

    tgn = oracle["model"]

    device = oracle["device"]



    nodes = np.asarray(nodes, dtype=np.int64)

    times = np.asarray(times, dtype=np.float32)



    if tgn.use_memory:

        memory = tgn.memory.get_memory(np.arange(tgn.n_nodes))

        last_update = tgn.memory.last_update



        time_diffs = torch.from_numpy(times).float().to(device) - last_update[nodes]



        if role == "source":

            time_diffs = (time_diffs - tgn.mean_time_shift_src) / tgn.std_time_shift_src

        else:

            time_diffs = (time_diffs - tgn.mean_time_shift_dst) / tgn.std_time_shift_dst

    else:

        memory = None

        time_diffs = None



    return tgn.embedding_module.compute_embedding(

        memory=memory,

        source_nodes=nodes,

        timestamps=times,

        n_layers=tgn.n_layers,

        n_neighbors=TGN_NUM_NEIGHBORS,

        time_diffs=time_diffs,

    )





def _score_tgn_pairs_no_update(oracle, pairs, t_cur, batch_size=512):

    tgn = oracle["model"]

    node_to_tgn = oracle["node_to_tgn"]



    out = []

    tgn.eval()



    mapped = []

    for u, v, new_len in pairs:

        if u not in node_to_tgn or v not in node_to_tgn:

            continue

        mapped.append((u, v, node_to_tgn[u], node_to_tgn[v], new_len))



    with torch.no_grad():

        for start in range(0, len(mapped), batch_size):

            batch = mapped[start:start + batch_size]

            if not batch:

                continue



            original_len = len(batch)



            if original_len == 1:

                batch = batch + batch



            src_nodes = np.asarray([x[2] for x in batch], dtype=np.int64)

            dst_nodes = np.asarray([x[3] for x in batch], dtype=np.int64)

            times = np.full(len(batch), float(t_cur), dtype=np.float32)



            src_emb = _compute_tgn_embeddings_no_update(

                oracle,

                src_nodes,

                times,

                role="source",

            )



            dst_emb = _compute_tgn_embeddings_no_update(

                oracle,

                dst_nodes,

                times,

                role="destination",

            )



            logits = tgn.affinity_score(src_emb, dst_emb).squeeze()

            probs = torch.sigmoid(logits).detach().cpu().numpy()

            probs = np.asarray(probs).reshape(-1)



            for item, p_hat in zip(batch[:original_len], probs[:original_len]):

                u_orig, v_orig, _, _, new_len = item

                out.append((u_orig, v_orig, new_len, float(p_hat)))



    return out





def _group_tests_by_src(tdf: pd.DataFrame):

    out = {}

    for row in tdf[["source", "destination", "prev_Path", "future_Path", "time"]].itertuples(index=False):

        src = int(row.source)

        out.setdefault(src, []).append(row)

    return out





def _build_static_graph_from_edges(edge_list):

    G = nx.DiGraph()

    for u, v, _, _ in edge_list:

        G.add_edge(int(u), int(v))

    return G





def _compute_node2vec_embeddings(

    edge_list,

    dimensions=128,

    walk_length=30,

    num_walks=200,

    p=1.0,

    q=1.0,

    workers=1,

    window=10,

    min_count=1,

    batch_words=256,

    seed=42,

):

    if Node2Vec is None:

        raise ImportError("The node2vec package is required for N2VLP-Static")

    G = _build_static_graph_from_edges(edge_list)



    if G.number_of_nodes() == 0:

        return {}



    node2vec = Node2Vec(

        G,

        dimensions=dimensions,

        walk_length=walk_length,

        num_walks=num_walks,

        p=p,

        q=q,

        workers=workers,

        seed=seed,

        quiet=True,

    )



    model = node2vec.fit(

        window=window,

        min_count=min_count,

        batch_words=batch_words,

    )



    embeddings = {}

    for node in G.nodes():

        embeddings[int(node)] = np.asarray(model.wv[str(node)], dtype=float)



    return embeddings





def _load_or_build_node2vec_embeddings(

    file_stem,

    prefix_name,

    edge_list,

    dimensions=128,

    walk_length=30,

    num_walks=200,

    p=1.0,

    q=1.0,

    workers=1,

    window=10,

    min_count=1,

    batch_words=256,

    seed=42,

    embeddings_dir=None,

    force_recompute=False,

):

    if embeddings_dir is None:

        embeddings_dir = EMBEDDINGS_DIR



    embeddings_dir = Path(embeddings_dir)

    embeddings_dir.mkdir(parents=True, exist_ok=True)



    cache_file = (

        embeddings_dir

        / f"{file_stem}_node2vec_{prefix_name}_d{dimensions}_wl{walk_length}_nw{num_walks}_p{p}_q{q}_seed{seed}.pkl"

    )



    if cache_file.exists() and not force_recompute:

        with cache_file.open("rb") as f:

            return pickle.load(f)



    embeddings = _compute_node2vec_embeddings(

        edge_list=edge_list,

        dimensions=dimensions,

        walk_length=walk_length,

        num_walks=num_walks,

        p=p,

        q=q,

        workers=workers,

        window=window,

        min_count=min_count,

        batch_words=batch_words,

        seed=seed,

    )



    with cache_file.open("wb") as f:

        pickle.dump(embeddings, f)



    return embeddings





def _load_embeddings_for_prefix(

    file_stem,

    prefix_name,

    edge_list,

    embeddings_dir=None,

    seed=42,

    force_recompute=False,

):

    return _load_or_build_node2vec_embeddings(

        file_stem=file_stem,

        prefix_name=prefix_name,

        edge_list=edge_list,

        dimensions=128,

        walk_length=30,

        num_walks=200,

        p=1.0,

        q=1.0,

        workers=1,

        window=10,

        min_count=1,

        batch_words=256,

        seed=seed,

        embeddings_dir=embeddings_dir,

        force_recompute=force_recompute,

    )



def _build_full_edge_list(df: pd.DataFrame) -> List[Tuple[int, int, int, float]]:

    edges = [

        (int(u), int(v), int(t), 1.0)

        for u, v, t in df[["source", "target", "time"]].itertuples(index=False)

    ]

    edges.sort(key=lambda x: x[2])

    return edges





def _normalize_df(csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(csv_path, sep=",")





    if {"u", "i", "ts_str"}.issubset(df.columns):

        df = df.rename(columns={"u": "source", "i": "target", "ts_str": "time"})



    if not {"source", "target", "time"}.issubset(df.columns):

        raise ValueError(f"{csv_path}: needs columns source,target,time")



    df = df[["source", "target", "time"]].copy()

    df["source"] = df["source"].astype(int)

    df["target"] = df["target"].astype(int)

    df["time"] = df["time"].astype(int)

    return df





def _load_missing_times(csv_stem: str) -> List[int]:

    mt_file = Path(f"{csv_stem}_missing_times.csv")

    if not mt_file.exists():

        return []

    mtdf = pd.read_csv(mt_file)

    if "Missing Times" not in mtdf.columns:

        return []

    return list(mtdf["Missing Times"].dropna().astype(int).unique())





def _build_streams(df: pd.DataFrame, pivot_time: int) -> Tuple[

    List[Tuple[int, int, int, float]],

    List[Tuple[int, int, int, float]],

    List[int],

    List[int],

    Tuple[int, int],

    Tuple[int, int],

]:

    edges = [

        (int(u), int(v), int(t), 1.0)

        for u, v, t in df[["source", "target", "time"]].itertuples(index=False)

    ]

    edges.sort(key=lambda x: x[2])



    vertex_set = set()

    pivot_edges = []

    for u, v, t, p in edges:

        if t <= pivot_time:

            vertex_set.add(u)

            vertex_set.add(v)

            pivot_edges.append((u, v, t, p))



    all_edges = [

        (u, v, t, p)

        for (u, v, t, p) in edges

        if u in vertex_set and v in vertex_set

    ]



    future_time_list = []

    seen_t = set()

    for _, _, t, _ in all_edges:

        if t > pivot_time and t not in seen_t:

            seen_t.add(t)

            future_time_list.append(t)



    tmin = min(df["time"])

    tmax = max(df["time"])

    return (

        all_edges,

        pivot_edges,

        sorted(list(vertex_set)),

        future_time_list,

        (tmin - 1, tmax + 1),

        (tmin - 1, pivot_time),

    )





def _build_global_activity_pool_from_prefix(edge_list, pool_size):

    if pool_size is None or int(pool_size) <= 0:

        return frozenset()



    activity = defaultdict(int)

    for u, v, _t, _p in edge_list:

        activity[int(u)] += 1

        activity[int(v)] += 1



    ranked = sorted(activity.items(), key=lambda item: (-int(item[1]), int(item[0])))

    return frozenset(int(node) for node, _score in ranked[:int(pool_size)])





def _bounded_reachable_nodes_from_prefix_adj(adj, source, max_distance):

    source = int(source)

    max_distance = int(max_distance)



    if max_distance <= 0:

        return set()



    visited = {source: 0}

    queue = [source]

    head = 0



    while head < len(queue):

        cur = queue[head]

        head += 1

        cur_dist = visited[cur]

        if cur_dist >= max_distance:

            continue



        for nxt in adj.get(cur, ()):

            nxt = int(nxt)

            if nxt in visited:

                continue

            visited[nxt] = cur_dist + 1

            queue.append(nxt)



    visited.pop(source, None)

    return set(visited.keys())





def _build_prefix_supported_landmark_pool(

    tests_df,

    edge_list,

    pool_size,

    connector_pool_size,

    active_fallback_pool=None,

    active_fallback_size=0,

):

    if not USE_PREFIX_SUPPORTED_LANDMARKS:

        return frozenset()



    if tests_df is None or len(tests_df) == 0:

        return frozenset()



    pool_size = max(0, int(pool_size or 0))

    if pool_size <= 0:

        return frozenset()





    adj = defaultdict(set)

    activity = defaultdict(int)

    for u, v, _t, _p in edge_list:

        u = int(u)

        v = int(v)

        adj[u].add(v)

        activity[u] += 1

        activity[v] += 1



    connector_support = defaultdict(int)

    prefix_path_support = defaultdict(int)

    reachability_support = defaultdict(int)

    destination_frequency = defaultdict(int)





    connector_pool_cache = {}

    for d in tests_df["destination"].astype(int).tolist():

        d = int(d)

        destination_frequency[d] += 1



        if d not in connector_pool_cache:

            connector_pool_cache[d] = _build_destination_connector_target_pool_from_prefix(

                edge_list=edge_list,

                dst=d,

                connector_pool_size=connector_pool_size,

                active_fallback_pool=active_fallback_pool,

                active_fallback_size=active_fallback_size,

            )



        for x in connector_pool_cache[d]:

            if int(x) == d:

                continue

            connector_support[int(x)] += 1





    if "prev_Path" in tests_df.columns:

        for path_str in tests_df["prev_Path"].tolist():

            nodes = _split_path_str(path_str)

            if len(nodes) <= 2:

                continue

            for x in set(int(z) for z in nodes[1:-1]):

                prefix_path_support[x] += 1





    if LANDMARK_REACHABILITY_WEIGHT != 0:

        max_dist = max(0, int(LANDMARK_REACHABILITY_MAX_DISTANCE or 0))

        for s in sorted(set(int(x) for x in tests_df["source"].astype(int).tolist())):

            for x in _bounded_reachable_nodes_from_prefix_adj(adj, s, max_dist):

                reachability_support[int(x)] += 1



    all_nodes = (

        set(connector_support.keys())

        | set(prefix_path_support.keys())

        | set(reachability_support.keys())

        | set(destination_frequency.keys())

    )



    scored = []

    for x in all_nodes:

        x = int(x)

        score = (

            float(LANDMARK_CONNECTOR_WEIGHT) * float(connector_support.get(x, 0))

            + float(LANDMARK_PREFIX_PATH_WEIGHT) * float(prefix_path_support.get(x, 0))

            + float(LANDMARK_REACHABILITY_WEIGHT) * float(reachability_support.get(x, 0))

            + float(LANDMARK_DESTINATION_FREQUENCY_WEIGHT) * float(destination_frequency.get(x, 0))

        )

        if score <= 0.0:

            continue

        scored.append((

            score,

            int(connector_support.get(x, 0)),

            int(prefix_path_support.get(x, 0)),

            int(reachability_support.get(x, 0)),

            int(destination_frequency.get(x, 0)),

            int(activity.get(x, 0)),

            x,

        ))



    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], -item[4], -item[5], item[6]))

    selected = [x for *_rest, x in scored[:pool_size]]

    return frozenset(int(x) for x in selected)





def _build_workload_landmark_pool_from_tests(

    tests_df,

    pool_size,

    edge_list=None,

    connector_pool_size=None,

    active_fallback_pool=None,

    active_fallback_size=0,

):

    if edge_list is None:





        if not USE_PREFIX_SUPPORTED_LANDMARKS or tests_df is None or len(tests_df) == 0:

            return frozenset()

        counts = defaultdict(int)

        for d in tests_df["destination"].astype(int).tolist():

            counts[int(d)] += 1

        ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), int(item[0])))

        return frozenset(int(node) for node, _count in ranked[:max(0, int(pool_size or 0))])



    return _build_prefix_supported_landmark_pool(

        tests_df=tests_df,

        edge_list=edge_list,

        pool_size=pool_size,

        connector_pool_size=connector_pool_size if connector_pool_size is not None else DEST_CONNECTOR_POOL_SIZE,

        active_fallback_pool=active_fallback_pool,

        active_fallback_size=active_fallback_size,

    )





def _union_destination_connector_pools_from_prefix(

    edge_list,

    destinations,

    connector_pool_size,

    active_fallback_pool=None,

    active_fallback_size=0,

):

    out = set()

    for d in sorted(set(int(x) for x in (destinations or []))):

        out.update(_build_destination_connector_target_pool_from_prefix(

            edge_list=edge_list,

            dst=d,

            connector_pool_size=connector_pool_size,

            active_fallback_pool=active_fallback_pool,

            active_fallback_size=active_fallback_size,

        ))

    return frozenset(out)





def _build_destination_connector_target_pool_from_prefix(

    edge_list,

    dst,

    connector_pool_size,

    active_fallback_pool=None,

    active_fallback_size=0,

):

    if not USE_DESTINATION_CONNECTOR_TARGETS:

        return frozenset()



    dst = int(dst)

    connector_pool_size = max(0, int(connector_pool_size or 0))

    active_fallback_size = max(0, int(active_fallback_size or 0))



    reverse_adj = defaultdict(set)

    activity = defaultdict(int)

    for u, v, _t, _p in edge_list:

        u = int(u)

        v = int(v)

        reverse_adj[v].add(u)

        activity[u] += 1

        activity[v] += 1





    visited = {dst: 0}

    queue = [dst]

    head = 0

    while head < len(queue):

        cur = queue[head]

        head += 1

        next_dist = visited[cur] + 1

        for pred in reverse_adj.get(cur, ()):

            pred = int(pred)

            if pred in visited:

                continue

            visited[pred] = next_dist

            queue.append(pred)



    ranked_connectors = sorted(

        ((node, dist) for node, dist in visited.items() if int(node) != dst),

        key=lambda item: (int(item[1]), -int(activity.get(int(item[0]), 0)), int(item[0])),

    )



    selected = []

    selected_set = set()



    for node, _dist in ranked_connectors[:connector_pool_size]:

        node = int(node)

        if node == dst or node in selected_set:

            continue

        selected.append(node)

        selected_set.add(node)



    if active_fallback_pool and active_fallback_size > 0:

        added = 0

        for node in active_fallback_pool:

            node = int(node)

            if node == dst or node in selected_set:

                continue

            selected.append(node)

            selected_set.add(node)

            added += 1

            if added >= active_fallback_size:

                break



    return frozenset(selected)





QUERY_TEST_FILES = {

    "enron": "enron.tsv",

    "email_eu": "email_eu.tsv",

    "email-eu": "email_eu.tsv",

    "collegemsg": "collegemsg.tsv",

    "bitcoin": "bitcoin.tsv",

    "ia-enron-employees_TimestampZero_NoDuplicate_sorted": "enron.tsv",

    "email-Eu-core-temporal_TimestampZero_NoDuplicate_sorted": "email_eu.tsv",

    "CollegeMsg_TimestampZero_NoDuplicate_sorted": "collegemsg.tsv",

    "ml_bitcoinotc_disperse_NoDuplicate_sorted": "bitcoin.tsv",

}





def _parse_paths_tsv(file_stem: str) -> pd.DataFrame:

    candidates = []

    simple_name = QUERY_TEST_FILES.get(file_stem)

    if simple_name is not None:

        candidates.append(TESTS_DIR / simple_name)

    if TOP100_TEST_SUFFIX:

        candidates.append(TESTS_DIR / f"{file_stem}{TOP100_TEST_SUFFIX}")

    candidates.append(TESTS_DIR / f"{file_stem}_Path_tests.tsv")



    path_tests = None

    for candidate in candidates:

        if candidate.exists():

            path_tests = candidate

            break

    if path_tests is None:

        checked = ", ".join(str(path) for path in candidates)

        raise FileNotFoundError(f"Missing test file for {file_stem}. Checked: {checked}")



    df = pd.read_csv(path_tests, sep="\t")

    exp_cols = {"source", "destination", "prev_Path", "future_Path", "time"}

    if not exp_cols.issubset(df.columns):

        raise ValueError(f"{path_tests} must contain columns: {sorted(exp_cols)}")

    return df





def _split_path_str(s: str) -> List[int]:

    if pd.isna(s) or s == "":

        return []

    return [int(x) for x in str(s).split(",") if x != ""]





def _extract_nodes_from_path_record(path_record):

    if isinstance(path_record, dict):

        return list(path_record.get("nodes", []))



    if isinstance(path_record, (tuple, list)) and len(path_record) >= 3:

        nodes = path_record[2]

        if isinstance(nodes, (list, tuple)):

            return list(nodes)



    return []





def _extract_nodes_from_path_record_view(path_record):

    if isinstance(path_record, dict):

        return path_record.get("nodes", ())



    if isinstance(path_record, (tuple, list)) and len(path_record) >= 3:

        nodes = path_record[2]

        if isinstance(nodes, (list, tuple)):

            return nodes



    return ()





def _candidate_sources_and_targets_from_global_paths(vertexToLv_dict, dst, reference_len):

    candidate_sources = []

    candidate_targets = {dst}



    for Lv in vertexToLv_dict.values():

        for rec in Lv:

            nodes = _extract_nodes_from_path_record_view(rec)



            if not nodes:

                continue



            candidate_targets.update(nodes)

            nodes_set = set(nodes)



            for depth, u in enumerate(nodes):

                if u == dst:

                    continue



                new_len = depth + 1



                if reference_len is None or new_len <= reference_len:

                    candidate_sources.append((u, new_len, nodes_set))





    return candidate_sources, candidate_targets





def _edge_score_cache_key(oracle_mode, u, v, t_cur):

    u = int(u)

    v = int(v)



    if oracle_mode == "static":

        return (u, v)



    return (u, v, int(t_cur))





def _edge_cache_tu_key(oracle_mode, u, t_cur, lp_edge_threshold):

    u = int(u)

    thr = float(lp_edge_threshold)



    if oracle_mode == "static":

        return ("static", u, thr)



    return (int(t_cur), u, thr)





def _store_edge_score_in_indexes(

    edge_score_cache,

    edge_cache_low_by_tu,

    edge_cache_high_by_tu,

    oracle_mode,

    u,

    v,

    t_cur,

    p_hat,

    lp_edge_threshold,

):

    u = int(u)

    v = int(v)

    p_hat = float(p_hat)



    cache_key = _edge_score_cache_key(oracle_mode, u, v, t_cur)

    edge_score_cache[cache_key] = p_hat



    tu_key = _edge_cache_tu_key(oracle_mode, u, t_cur, lp_edge_threshold)



    if p_hat >= lp_edge_threshold:

        edge_cache_high_by_tu[tu_key][v] = p_hat

        edge_cache_low_by_tu[tu_key].discard(v)

    else:

        edge_cache_low_by_tu[tu_key].add(v)

        edge_cache_high_by_tu[tu_key].pop(v, None)





def _freeze_for_path_cache_key(value):

    if isinstance(value, dict):

        return tuple(

            sorted(

                (

                    _freeze_for_path_cache_key(k),

                    _freeze_for_path_cache_key(v),

                )

                for k, v in value.items()

            )

        )



    if isinstance(value, (list, tuple)):

        return tuple(_freeze_for_path_cache_key(x) for x in value)



    if isinstance(value, set):

        return tuple(sorted(_freeze_for_path_cache_key(x) for x in value))



    if isinstance(value, np.generic):

        return value.item()



    try:

        hash(value)

        return value

    except TypeError:

        return repr(value)





def _path_label_cache_key(path_record):

    return _freeze_for_path_cache_key(path_record)





def _init_incremental_path_candidate_cache(cache_state):

    if cache_state is None:

        return None



    cache_state.setdefault("entries", {})

    cache_state.setdefault("path_ids", {})

    cache_state.setdefault("next_lv_path_id", 0)

    cache_state.setdefault("last_active_entry_keys", set())

    return cache_state





def _compile_candidate_records_for_path(path_record, path_key, dst, reference_len, lv_path_id):

    dst = int(dst)

    nodes = _extract_nodes_from_path_record_view(path_record)



    if not nodes:

        return None



    nodes_tuple = tuple(int(x) for x in nodes)

    if len(nodes_tuple) == 0:

        return None



    nodes_set = frozenset(nodes_tuple)

    records = []

    seen_sources_in_path = set()



    for depth, u in enumerate(nodes_tuple):

        u = int(u)



        if u == dst:

            continue



        if u in seen_sources_in_path:

            continue

        seen_sources_in_path.add(u)



        new_len = int(depth + 1)



        if reference_len is not None and new_len > reference_len:

            continue



        records.append((u, int(lv_path_id), new_len, nodes_set))



    return {

        "path_key": path_key,

        "lv_path_id": int(lv_path_id),

        "nodes_tuple": nodes_tuple,

        "nodes_set": nodes_set,

        "records": records,

    }





def _get_compiled_candidate_path_index(

    vertexToLv_dict,

    dst,

    reference_len,

    compiled_candidate_index_cache,

    path_state_version=None,

):

    _candidate_path_index_total_start = time_lib.perf_counter()



    dst = int(dst)

    ref_key = -1 if reference_len is None else int(reference_len)



    cache_state = _init_incremental_path_candidate_cache(compiled_candidate_index_cache)





    if cache_state is None:

        cache_state = _init_incremental_path_candidate_cache({})



    _cache_lookup_start = time_lib.perf_counter()

    active_entry_keys = []

    seen_active_entry_keys = set()

    active_path_record_info = []



    for Lv in vertexToLv_dict.values():

        for rec in Lv:

            nodes = _extract_nodes_from_path_record_view(rec)

            if not nodes:

                continue



            path_key = _path_label_cache_key(rec)

            entry_key = (ref_key, path_key)





            if entry_key in seen_active_entry_keys:

                continue



            seen_active_entry_keys.add(entry_key)

            active_entry_keys.append(entry_key)

            active_path_record_info.append((entry_key, path_key, rec))



    active_entry_key_set = set(active_entry_keys)

    previous_active_keys = set(cache_state.get("last_active_entry_keys", set()))

    candidate_path_index_cache_lookup_s = time_lib.perf_counter() - _cache_lookup_start



    entries = cache_state["entries"]

    path_ids = cache_state["path_ids"]



    removed_entry_keys = previous_active_keys - active_entry_key_set

    added_or_missing_entry_keys = [

        entry_key

        for entry_key, _, _ in active_path_record_info

        if entry_key not in entries

    ]

    unchanged_entry_keys = active_entry_key_set - set(added_or_missing_entry_keys)



    _candidate_path_index_build_start = time_lib.perf_counter()



    for entry_key in removed_entry_keys:

        entries.pop(entry_key, None)



    added_count = 0

    for entry_key, path_key, rec in active_path_record_info:

        if entry_key in entries:

            continue



        if path_key not in path_ids:

            path_ids[path_key] = int(cache_state["next_lv_path_id"])

            cache_state["next_lv_path_id"] += 1



        entry = _compile_candidate_records_for_path(

            path_record=rec,

            path_key=path_key,

            dst=dst,

            reference_len=reference_len,

            lv_path_id=path_ids[path_key],

        )



        if entry is not None:

            entries[entry_key] = entry

            added_count += 1



    candidate_path_index_build_s = time_lib.perf_counter() - _candidate_path_index_build_start



    _candidate_path_index_store_start = time_lib.perf_counter()

    cache_state["last_active_entry_keys"] = active_entry_key_set

    candidate_path_index_cache_store_s = time_lib.perf_counter() - _candidate_path_index_store_start



    records_by_source = defaultdict(list)

    target_refcount = defaultdict(int)

    num_active_entries = 0



    for entry_key in active_entry_keys:

        entry = entries.get(entry_key)

        if entry is None:

            continue



        num_active_entries += 1

        for node in entry["nodes_tuple"]:

            target_refcount[int(node)] += 1



        for u, lv_path_id, new_len, nodes_set in entry["records"]:

            records_by_source[int(u)].append((int(lv_path_id), int(new_len), nodes_set))



    candidate_targets = set(target_refcount.keys())

    candidate_targets.add(dst)



    for u in records_by_source:

        records_by_source[u].sort(key=lambda x: (x[1], x[0]))



    cache_hit = int(

        len(added_or_missing_entry_keys) == 0

        and len(removed_entry_keys) == 0

        and active_entry_key_set == previous_active_keys

    )



    stats = {

        "num_candidate_source_records": sum(len(x) for x in records_by_source.values()),

        "num_candidate_sources": len(records_by_source),

        "num_candidate_targets": len(candidate_targets),

        "num_lv_paths": num_active_entries,

        "candidate_path_index_cache_hit": cache_hit,

        "candidate_path_index_cache_size": len(entries),

        "candidate_path_index_cache_lookup_s": candidate_path_index_cache_lookup_s,

        "candidate_path_index_build_s": candidate_path_index_build_s,

        "candidate_path_index_cache_store_s": candidate_path_index_cache_store_s,

        "candidate_path_index_cache_update_s": candidate_path_index_build_s + candidate_path_index_cache_store_s,

        "candidate_path_index_total_s": time_lib.perf_counter() - _candidate_path_index_total_start,

        "path_candidate_cache_active": num_active_entries,

        "path_candidate_cache_hits": len(unchanged_entry_keys),

        "path_candidate_cache_misses": added_count,

        "path_candidate_cache_added": added_count,

        "path_candidate_cache_removed": len(removed_entry_keys),

        "path_candidate_cache_unchanged": len(unchanged_entry_keys),

    }



    return records_by_source, frozenset(candidate_targets), stats





def _build_candidate_pairs_fast_with_cache(

    vertexToLv_dict,

    dst,

    reference_len,

    observed_out,

    accepted_out,

    t_cur,

    edge_score_cache,

    edge_cache_low_by_tu,

    edge_cache_high_by_tu,

    lp_edge_threshold,

    oracle_mode,

    compiled_candidate_index_cache=None,

    path_state_version=0,

    use_edge_score_cache=True,

    use_candidate_path_index_cache=True,

    active_target_pool=None,

    landmark_target_pool=None,

    landmark_connector_target_pool=None,

    landmark_nodes=None,

    root_source=None,

):

    dst = int(dst)



    active_candidate_index_cache = (

        compiled_candidate_index_cache

        if use_candidate_path_index_cache

        else None

    )



    (

        records_by_source,

        candidate_targets,

        index_stats,

    ) = _get_compiled_candidate_path_index(

        vertexToLv_dict=vertexToLv_dict,

        dst=dst,

        reference_len=reference_len,

        compiled_candidate_index_cache=active_candidate_index_cache,

        path_state_version=path_state_version,

    )



    missing_candidate_pair_to_best_len = {}

    missing_candidate_occurrences_by_pair = defaultdict(list)

    cached_scored_candidate_occurrences = []



    cache_hits = 0

    cache_misses = 0

    cache_low_skips = 0

    cache_high_reused = 0

    num_candidate_occurrences_total = 0

    unique_candidate_pairs = set()



    path_candidate_targets_set = set(candidate_targets)

    connector_target_pool_set = set(active_target_pool or ())

    landmark_nodes_set = set(int(x) for x in (landmark_nodes or ()))

    landmark_target_pool_set = set(int(x) for x in (landmark_target_pool or ()))

    landmark_connector_target_pool_set = set(int(x) for x in (landmark_connector_target_pool or ()))





    if USE_DESTINATION_CONNECTOR_TARGETS and (connector_target_pool_set or landmark_target_pool_set or landmark_connector_target_pool_set or landmark_nodes_set):

        source_min_lens = []

        for src_u, src_records in records_by_source.items():

            if not src_records:

                continue

            min_new_len = min(int(new_len) for _lv_path_id, new_len, _nodes_set in src_records)

            source_min_lens.append((min_new_len, int(src_u)))



        source_min_lens.sort(key=lambda x: (x[0], x[1]))

        connector_source_budget = max(0, int(DEST_CONNECTOR_SOURCE_BUDGET))

        shallow_sources_set = {

            int(src_u)

            for _min_len, src_u in source_min_lens[:connector_source_budget]

        }



        all_record_sources_set = {int(src_u) for src_u in records_by_source.keys()}

        connector_sources_set = set()



        if INCLUDE_CONNECTOR_SOURCES_AS_CANDIDATE_SOURCES:

            connector_sources_set |= all_record_sources_set & connector_target_pool_set



        if INCLUDE_LANDMARK_CONNECTOR_SOURCES:

            connector_sources_set |= all_record_sources_set & landmark_connector_target_pool_set



        if INCLUDE_REACHED_LANDMARK_SOURCES:

            connector_sources_set |= all_record_sources_set & landmark_nodes_set



        active_target_sources_set = shallow_sources_set | connector_sources_set



        base_connector_targets_set = (

            {dst}

            | connector_target_pool_set

            | landmark_target_pool_set

            | landmark_connector_target_pool_set

            | landmark_nodes_set

        )

        if INCLUDE_PATH_TARGETS_IN_CONNECTOR_MODE:

            base_connector_targets_set |= path_candidate_targets_set

        candidate_targets_set = set(base_connector_targets_set)

    else:



        shallow_sources_set = {int(src_u) for src_u in records_by_source.keys()}

        connector_sources_set = set()

        active_target_sources_set = {int(src_u) for src_u in records_by_source.keys()}

        candidate_targets_set = path_candidate_targets_set



    completion_only_sources_set = set()



    for u, records in records_by_source.items():

        u = int(u)



        if USE_DESTINATION_CONNECTOR_TARGETS and (connector_target_pool_set or landmark_target_pool_set or landmark_connector_target_pool_set or landmark_nodes_set):

            if u in active_target_sources_set:

                candidate_targets_set_for_source = candidate_targets_set

            elif INCLUDE_REACHED_NODES_AS_COMPLETION_SOURCES:





                candidate_targets_set_for_source = {dst}

                completion_only_sources_set.add(u)

            else:

                continue

        else:

            candidate_targets_set_for_source = path_candidate_targets_set



        blocked = set(observed_out.get(u, ()))



        if BLOCK_ACCEPTED_PREDICTED_EDGES:

            blocked.update(accepted_out.get(u, ()))



        blocked.add(u)



        if use_edge_score_cache:

            tu_key = _edge_cache_tu_key(oracle_mode, u, t_cur, lp_edge_threshold)

            low_set = edge_cache_low_by_tu.get(tu_key, set())

            high_dict = edge_cache_high_by_tu.get(tu_key, {})

            high_set = set(high_dict.keys())

            seen_score_targets = low_set | high_set

        else:

            low_set = set()

            high_dict = {}

            high_set = set()

            seen_score_targets = set()



        for lv_path_id, new_len, nodes_set in records:

            lv_path_id = int(lv_path_id)

            new_len = int(new_len)



            if reference_len is None:

                allowed_targets = candidate_targets_set_for_source

            else:

                if new_len < reference_len:

                    allowed_targets = candidate_targets_set_for_source

                elif new_len == reference_len:

                    allowed_targets = {dst}

                else:

                    continue



            valid_targets = set(allowed_targets)

            valid_targets.difference_update(blocked)

            valid_targets.difference_update(nodes_set)





            if DISABLE_ROOT_DIRECT_EDGES and (int(new_len) == 1 or (root_source is not None and int(u) == int(root_source) and int(new_len) <= 1)):

                valid_targets.discard(dst)



            if not valid_targets:

                continue



            cached_high_targets = valid_targets & high_set

            cached_low_targets = valid_targets & low_set

            missing_targets = valid_targets - seen_score_targets



            for v in cached_high_targets:

                v = int(v)

                p_hat = edge_score_cache.get(_edge_score_cache_key(oracle_mode, u, v, t_cur))

                if p_hat is None:

                    p_hat = high_dict[v]

                cached_scored_candidate_occurrences.append(

                    (lv_path_id, u, v, new_len, float(p_hat))

                )

                unique_candidate_pairs.add((u, v))



            for v in missing_targets:

                v = int(v)

                pair_key = (u, v)

                old_len = missing_candidate_pair_to_best_len.get(pair_key)

                if old_len is None or new_len < old_len:

                    missing_candidate_pair_to_best_len[pair_key] = new_len

                missing_candidate_occurrences_by_pair[pair_key].append(

                    (lv_path_id, u, v, new_len)

                )

                unique_candidate_pairs.add(pair_key)



            for v in cached_low_targets:

                unique_candidate_pairs.add((u, int(v)))



            num_candidate_occurrences_total += (

                len(cached_high_targets) + len(cached_low_targets) + len(missing_targets)

            )

            cache_high_reused += len(cached_high_targets)

            cache_low_skips += len(cached_low_targets)

            cache_hits += len(cached_high_targets) + len(cached_low_targets)

            cache_misses += len(missing_targets)



    stats = {

        "num_candidate_source_records": index_stats.get("num_candidate_source_records", 0),

        "num_candidate_sources": index_stats.get("num_candidate_sources", 0),

        "num_candidate_targets_from_paths": len(path_candidate_targets_set),

        "num_active_target_pool": len(connector_target_pool_set),

        "num_destination_connector_targets": len(connector_target_pool_set),

        "num_landmark_targets": len(landmark_nodes_set),

        "num_landmark_connector_targets": len(landmark_connector_target_pool_set),

        "num_active_target_sources": len(active_target_sources_set),

        "num_shallow_candidate_sources": len(shallow_sources_set),

        "num_connector_candidate_sources": len(connector_sources_set),

        "include_reached_nodes_as_completion_sources": int(INCLUDE_REACHED_NODES_AS_COMPLETION_SOURCES),

        "num_completion_only_candidate_sources": len(completion_only_sources_set),

        "active_target_source_budget": int(DEST_CONNECTOR_SOURCE_BUDGET),

        "destination_connector_source_budget": int(DEST_CONNECTOR_SOURCE_BUDGET),

        "use_active_targets_only_for_top_sources": int(USE_ACTIVE_TARGETS_ONLY_FOR_TOP_SOURCES),

        "use_destination_connector_targets": int(USE_DESTINATION_CONNECTOR_TARGETS),

        "include_path_targets_in_connector_mode": int(INCLUDE_PATH_TARGETS_IN_CONNECTOR_MODE),

        "num_candidate_targets_total": len(candidate_targets_set),

        "num_candidate_targets": len(candidate_targets_set),

        "num_lv_paths": index_stats.get("num_lv_paths", 0),

        "num_candidate_pairs_total": num_candidate_occurrences_total,

        "num_unique_candidate_pairs": len(unique_candidate_pairs),

        "num_missing_candidates_to_score": len(missing_candidate_pair_to_best_len),

        "num_missing_candidate_occurrences": sum(len(v) for v in missing_candidate_occurrences_by_pair.values()),

        "num_cached_scored_candidates": len(cached_scored_candidate_occurrences),

        "candidate_cache_hits": cache_hits,

        "candidate_cache_misses": cache_misses,

        "candidate_cache_low_skips": cache_low_skips,

        "candidate_cache_high_reused": cache_high_reused,

        "use_edge_score_cache": int(use_edge_score_cache),

        "use_candidate_path_index_cache": int(use_candidate_path_index_cache),

        "edge_score_cache_size": len(edge_score_cache) if use_edge_score_cache else 0,

        "edge_cache_low_index_size": sum(len(v) for v in edge_cache_low_by_tu.values()) if use_edge_score_cache else 0,

        "edge_cache_high_index_size": sum(len(v) for v in edge_cache_high_by_tu.values()) if use_edge_score_cache else 0,

        "candidate_path_index_cache_hit": index_stats.get("candidate_path_index_cache_hit", 0) if use_candidate_path_index_cache else 0,

        "candidate_path_index_cache_size": index_stats.get("candidate_path_index_cache_size", 0) if use_candidate_path_index_cache else 0,

        "candidate_path_index_cache_lookup_s": index_stats.get("candidate_path_index_cache_lookup_s", 0.0),

        "candidate_path_index_build_s": index_stats.get("candidate_path_index_build_s", 0.0),

        "candidate_path_index_cache_store_s": index_stats.get("candidate_path_index_cache_store_s", 0.0),

        "candidate_path_index_cache_update_s": index_stats.get("candidate_path_index_cache_update_s", 0.0) if use_candidate_path_index_cache else 0.0,

        "candidate_path_index_total_s": index_stats.get("candidate_path_index_total_s", 0.0),

        "path_candidate_cache_active": index_stats.get("path_candidate_cache_active", 0),

        "path_candidate_cache_hits": index_stats.get("path_candidate_cache_hits", 0),

        "path_candidate_cache_misses": index_stats.get("path_candidate_cache_misses", 0),

        "path_candidate_cache_added": index_stats.get("path_candidate_cache_added", 0),

        "path_candidate_cache_removed": index_stats.get("path_candidate_cache_removed", 0),

        "path_candidate_cache_unchanged": index_stats.get("path_candidate_cache_unchanged", 0),

    }



    return (

        missing_candidate_pair_to_best_len,

        cached_scored_candidate_occurrences,

        missing_candidate_occurrences_by_pair,

        stats,

    )



def _candidate_predecessors_from_dst_skyline(destination_paths_dict, shortest_dict, dst, vertex_list, reference_len):

    candidate_info = {}

    path_list = destination_paths_dict.get(dst, [])



    for path_record in path_list:

        nodes = _extract_nodes_from_path_record(path_record)

        if len(nodes) < 2:

            continue



        for depth, x in enumerate(nodes[:-1]):

            if x == dst:

                continue



            cand_len = depth + 1



            if reference_len is None or cand_len <= reference_len:

                if x not in candidate_info or cand_len < candidate_info[x]:

                    candidate_info[x] = cand_len



    if candidate_info:

        return candidate_info



    return _candidate_predecessors_for_dst(shortest_dict, dst, vertex_list, reference_len)





def old_candidate_predecessors_from_global_paths(vertexToLv_dict, shortest_dict, dst, vertex_list, reference_len):

    candidate_info = {}



    for Lv in vertexToLv_dict.values():

        for rec in Lv:

            if isinstance(rec, dict):

                nodes = list(rec.get("nodes", []))

            else:

                nodes = _extract_nodes_from_path_record(rec)



            if not nodes:

                continue



            for depth, x in enumerate(nodes):

                if x == dst:

                    continue



                cand_len = depth + 1



                if reference_len is None or cand_len <= reference_len:

                    if x not in candidate_info or cand_len < candidate_info[x]:

                        candidate_info[x] = cand_len



    if candidate_info:

        return candidate_info



    return _candidate_predecessors_for_dst(shortest_dict, dst, vertex_list, reference_len)





def _candidate_predecessors_for_dst(shortest_dict, dst, vertex_list, reference_len):

    candidate_info = {}



    for x in vertex_list:

        if x == dst:

            continue



        x_len = _current_best_len(shortest_dict, x)

        if x_len == math.inf:

            continue



        cand_len = x_len + 1



        if reference_len is None or cand_len <= reference_len:

            candidate_info[x] = cand_len



    return candidate_info





def _current_best_len(shortest_dict, node):

    entry = shortest_dict.get(node, (math.inf, [], 0, 0.0))

    d = entry[0]

    return d if d is not None else math.inf





def _next_future_timestamp_after(future_times, t_cur):

    if not future_times:

        return None



    idx = bisect_right(future_times, int(t_cur))

    if idx >= len(future_times):

        return None

    return int(future_times[idx])





def _accepted_candidate_topk_key(candidate):

    lv_path_id, u, v, new_len, p_hat = candidate

    return (

        int(new_len),

        -float(p_hat),

        int(u),

        int(v),

    )





def _score_one_hop_bridge_scores_for_topk(

    oracle,

    accepted_candidates_before_lv_path_cap,

    dst,

    t_bridge,

    pivot_time_inference,

    embeddings_dict,

    edge_score_cache,

    edge_cache_low_by_tu,

    edge_cache_high_by_tu,

    lp_edge_threshold,

    oracle_mode,

    use_edge_score_cache=True,

):

    dst = int(dst)

    if t_bridge is None:

        return {}, {

            "num_bridge_probe_pairs": 0,

            "num_bridge_probe_cache_hits": 0,

            "num_bridge_probe_scored": 0,

            "bridge_probe_s": 0.0,

        }



    start_s = time_lib.perf_counter()



    bridge_sources = sorted({

        int(v)

        for (_, _, v, _, _) in accepted_candidates_before_lv_path_cap

        if int(v) != dst

    })



    bridge_scores = {}

    missing_bridge_pair_to_len = {}

    cache_hits = 0



    for v in bridge_sources:

        cache_key = _edge_score_cache_key(oracle_mode, v, dst, t_bridge)

        if use_edge_score_cache and cache_key in edge_score_cache:

            bridge_scores[v] = float(edge_score_cache[cache_key])

            cache_hits += 1

        else:

            missing_bridge_pair_to_len[(v, dst)] = 1



    scored = []

    if missing_bridge_pair_to_len:

        scored = _score_candidate_pairs_batch(

            oracle=oracle,

            candidate_pair_to_best_len=missing_bridge_pair_to_len,

            t_cur=t_bridge,

            pivot_time_inference=pivot_time_inference,

            embeddings_dict=embeddings_dict,

        )



        for u_bridge, v_bridge, _new_len, p_hat in scored:

            u_bridge = int(u_bridge)

            v_bridge = int(v_bridge)

            if v_bridge != dst:

                continue



            bridge_scores[u_bridge] = float(p_hat)



            if use_edge_score_cache:

                _store_edge_score_in_indexes(

                    edge_score_cache=edge_score_cache,

                    edge_cache_low_by_tu=edge_cache_low_by_tu,

                    edge_cache_high_by_tu=edge_cache_high_by_tu,

                    oracle_mode=oracle_mode,

                    u=u_bridge,

                    v=v_bridge,

                    t_cur=t_bridge,

                    p_hat=p_hat,

                    lp_edge_threshold=lp_edge_threshold,

                )



    return bridge_scores, {

        "num_bridge_probe_pairs": len(bridge_sources),

        "num_bridge_probe_cache_hits": cache_hits,

        "num_bridge_probe_scored": len(scored),

        "bridge_probe_s": time_lib.perf_counter() - start_s,

    }





def _accepted_candidate_shortcut_bridge_key(candidate, dst, bridge_scores, bridge_threshold):

    lv_path_id, u, v, new_len, p_uv = candidate

    dst = int(dst)

    u = int(u)

    v = int(v)

    new_len = int(new_len)

    p_uv = float(p_uv)

    bridge_threshold = float(bridge_threshold)





    if v == dst:

        estimated_shortcut_len = new_len

        shortcut_type_rank = 0

        shortcut_score = p_uv

    else:

        p_bridge = float(bridge_scores.get(v, 0.0))

        if p_bridge >= bridge_threshold:

            estimated_shortcut_len = new_len + 1

            shortcut_type_rank = 1

            shortcut_score = p_uv * p_bridge

        else:

            estimated_shortcut_len = 10**12

            shortcut_type_rank = 2

            shortcut_score = p_uv



    return (

        int(estimated_shortcut_len),

        int(new_len),

        int(shortcut_type_rank),

        -float(shortcut_score),

        -float(p_uv),

        int(u),

        int(v),

    )





def _is_root_direct_candidate(candidate, dst):

    _lv_path_id, _u, v, new_len, _p_uv = candidate

    return int(v) == int(dst) and int(new_len) == 1





def _candidate_estimated_total_len(candidate, dst):

    _lv_path_id, _u, v, new_len, _p_uv = candidate

    v = int(v)

    dst = int(dst)

    new_len = int(new_len)



    if v == dst:

        return new_len



    return new_len + 1





def _candidate_shortcut_gain(candidate, dst, reference_len):

    if reference_len is None or reference_len == math.inf:

        return 0



    estimated_total_len = _candidate_estimated_total_len(candidate, dst)

    return int(reference_len) - int(estimated_total_len)





def _accepted_candidate_shortcut_gain_key(candidate, dst, reference_len):

    lv_path_id, u, v, new_len, p_uv = candidate

    dst = int(dst)

    u = int(u)

    v = int(v)

    new_len = int(new_len)

    p_uv = float(p_uv)



    estimated_total_len = _candidate_estimated_total_len(candidate, dst)

    shortcut_gain = _candidate_shortcut_gain(candidate, dst, reference_len)

    is_root_direct = _is_root_direct_candidate(candidate, dst)



    if is_root_direct and DEMOTE_ROOT_DIRECT_EDGES:

        bucket = 3

    elif shortcut_gain > 0:

        bucket = 0

    elif shortcut_gain == 0:

        bucket = 1

    else:

        bucket = 2



    return (

        int(bucket),

        int(estimated_total_len),

        -int(shortcut_gain),

        int(new_len),

        -float(p_uv),

        int(u),

        int(v),

    )





def _compute_target_support_for_topk(accepted_candidates, dst, reference_len):

    dst = int(dst)

    target_support = defaultdict(float)



    for cand in accepted_candidates:

        _lv_path_id, _u, v, _new_len, p_hat = cand

        v = int(v)



        if v == dst:

            continue



        shortcut_gain = _candidate_shortcut_gain(cand, dst, reference_len)

        if shortcut_gain <= 0:

            continue



        target_support[v] += float(shortcut_gain) * float(p_hat)



    return target_support





def _accepted_candidate_target_support_key(candidate, dst, reference_len, target_support):

    _lv_path_id, u, v, new_len, p_uv = candidate

    dst = int(dst)

    u = int(u)

    v = int(v)

    new_len = int(new_len)

    p_uv = float(p_uv)



    estimated_total_len = _candidate_estimated_total_len(candidate, dst)

    shortcut_gain = _candidate_shortcut_gain(candidate, dst, reference_len)

    is_root_direct = _is_root_direct_candidate(candidate, dst)



    if v != dst:

        support_score = float(target_support.get(v, 0.0))

    else:

        support_score = float(target_support.get(u, 0.0))



    if is_root_direct and DEMOTE_ROOT_DIRECT_EDGES:

        bucket = 5

    elif shortcut_gain > 0 and v != dst:

        bucket = 0

    elif shortcut_gain > 0 and v == dst and support_score > 0.0 and not is_root_direct:

        bucket = 1

    elif shortcut_gain > 0:

        bucket = 2

    elif shortcut_gain == 0:

        bucket = 3

    else:

        bucket = 4



    return (

        int(bucket),

        int(estimated_total_len),

        -int(shortcut_gain),

        -float(support_score),

        int(new_len),

        -float(p_uv),

        int(u),

        int(v),

    )





import argparse

import inspect

import re

import types

import zipfile

from types import SimpleNamespace

from typing import Any, Iterable, Optional, Sequence

from collections import Counter



import torch.nn as nn

import torch.nn.functional as F





JODIE_ZIP_PATH = BASE_DIR / "External" / "jodie-master.zip"

JODIE_REPO_DIR = BASE_DIR / "External" / "jodie"

if not JODIE_REPO_DIR.exists() and JODIE_ZIP_PATH.exists():

    with zipfile.ZipFile(JODIE_ZIP_PATH, "r") as zf:

        zf.extractall(BASE_DIR / "External")





try:

    import gpustat

except ModuleNotFoundError:

    sys.modules["gpustat"] = types.ModuleType("gpustat")



try:

    import tqdm

except ModuleNotFoundError:

    tqdm_stub = types.ModuleType("tqdm")



    def _identity_tqdm(x=None, *args, **kwargs):

        return x if x is not None else []



    def _identity_trange(*args, **kwargs):

        return range(*args)



    tqdm_stub.tqdm = _identity_tqdm

    tqdm_stub.trange = _identity_trange

    tqdm_stub.tqdm_notebook = _identity_tqdm

    tqdm_stub.tnrange = _identity_trange

    sys.modules["tqdm"] = tqdm_stub

try:

    import tqdm

except ModuleNotFoundError:

    _tqdm_stub = types.ModuleType("tqdm")

    def _identity_tqdm(x=None, *args, **kwargs):

        return x if x is not None else []

    def _identity_trange(*args, **kwargs):

        return range(*args)

    _tqdm_stub.tqdm = _identity_tqdm

    _tqdm_stub.trange = _identity_trange

    _tqdm_stub.tqdm_notebook = _identity_tqdm

    _tqdm_stub.tnrange = _identity_trange

    sys.modules["tqdm"] = _tqdm_stub

if JODIE_REPO_DIR.exists():

    sys.path.insert(0, str(JODIE_REPO_DIR))

try:

    from library_models import JODIE as OriginalJODIE

except ModuleNotFoundError as exc:

    raise ImportError(

        "Could not import JODIE from External/jodie/library_models.py. "

        "Place External/jodie/ next to this module."

    ) from exc



JODIE_FLUSH_FINAL_TBATCH = True

JODIE_EVAL_TOPK_PER_SOURCE = 100

JODIE_PATH_TOPK_PER_SOURCE = 10

JODIE_EPOCHS = 50

JODIE_EMBEDDING_DIM = 128

JODIE_LR = 1e-3

JODIE_WEIGHT_DECAY = 1e-5

JODIE_ONLINE_EVAL_UPDATE_TRUE_EDGES = True

JODIE_PATH_STATE_MODE = "self_update"

JODIE_SELF_UPDATE_TOP_RANK = 1

JODIE_SELF_UPDATE_ORDER = "score_desc"

JODIE_RR_SCORE_MODE = "reciprocal_rank"

RANK_METRIC_CUTOFFS = (1, 5, 10, 20, 50, 100)

MISSING_TRUE_EDGE_SCORE = 0.0





def _print_score_summary(y_true, y_score, label="scores"):

    if not y_true:

        return



    pos_scores = [float(s) for y, s in zip(y_true, y_score) if y == 1]

    neg_scores = [float(s) for y, s in zip(y_true, y_score) if y == 0]



    if len(pos_scores) == 0 or len(neg_scores) == 0:

        print(f"[{label} score summary] pos_n={len(pos_scores)} neg_n={len(neg_scores)}")

        return



    auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) >= 2 else np.nan

    flipped_auc = (

        float(roc_auc_score(y_true, [1.0 - float(s) for s in y_score]))

        if len(set(y_true)) >= 2

        else np.nan

    )



    print(

        f"[{label} score summary] "

        f"pos_n={len(pos_scores)} neg_n={len(neg_scores)} "

        f"pos_mean={np.mean(pos_scores):.6f} neg_mean={np.mean(neg_scores):.6f} "

        f"pos_median={np.median(pos_scores):.6f} neg_median={np.median(neg_scores):.6f} "

        f"pos_q90={np.quantile(pos_scores, 0.90):.6f} neg_q90={np.quantile(neg_scores, 0.90):.6f} "

        f"auc={auc:.6f} flipped_auc={flipped_auc:.6f}"

    )





def _threshold_table_from_scores(y_true, y_score, lp_edge_thresholds, label="LP"):

    _print_score_summary(y_true, y_score, label=label)



    if len(set(y_true)) < 2:

        auc = np.nan

        ap = np.nan

    else:

        auc = float(roc_auc_score(y_true, y_score))

        ap = float(average_precision_score(y_true, y_score))



    rows = []

    best_threshold = lp_edge_thresholds[0]

    best_sum = -1.0



    for thr in lp_edge_thresholds:

        tp = sum(1 for y, s in zip(y_true, y_score) if y == 1 and s >= thr)

        fp = sum(1 for y, s in zip(y_true, y_score) if y == 0 and s >= thr)

        fn = sum(1 for y, s in zip(y_true, y_score) if y == 1 and s < thr)

        tn = sum(1 for y, s in zip(y_true, y_score) if y == 0 and s < thr)



        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        acc = (tp + tn) / max(1, tp + tn + fp + fn)



        rows.append(

            {

                "threshold": thr,

                "tp": tp,

                "fp": fp,

                "fn": fn,

                "tn": tn,

                "precision": precision,

                "recall": recall,

                "f1": f1,

                "accuracy": acc,

                "auc": auc,

                "ap": ap,

            }

        )



        score = precision + recall + acc

        if score > best_sum:

            best_sum = score

            best_threshold = thr



    return best_threshold, pd.DataFrame(rows)





def _rank_metric_dict(prefix, ranks, total_true_edges, topk_per_source):

    total_true_edges = int(total_true_edges)

    ranks = np.asarray(ranks, dtype=np.float64)



    if total_true_edges <= 0:

        out = {

            f"{prefix}_total_true_edges": 0,

            f"{prefix}_hits_at_topk": 0,

            f"{prefix}_topk_recall": np.nan,

            f"{prefix}_mrr": np.nan,

            f"{prefix}_mean_hit_rank": np.nan,

            f"{prefix}_median_hit_rank": np.nan,

        }

        for k in RANK_METRIC_CUTOFFS:

            out[f"{prefix}_hits_at_{k}"] = 0

            out[f"{prefix}_hits_at_{k}_rate"] = np.nan

        return out



    finite_mask = np.isfinite(ranks)

    finite_ranks = ranks[finite_mask]

    hits_at_topk = int(np.sum(finite_mask))

    rr_sum = float(np.sum(1.0 / finite_ranks)) if finite_ranks.size > 0 else 0.0



    out = {

        f"{prefix}_total_true_edges": total_true_edges,

        f"{prefix}_hits_at_topk": hits_at_topk,

        f"{prefix}_topk_recall": hits_at_topk / total_true_edges,

        f"{prefix}_mrr": rr_sum / total_true_edges,

        f"{prefix}_mean_hit_rank": float(np.mean(finite_ranks)) if finite_ranks.size > 0 else np.nan,

        f"{prefix}_median_hit_rank": float(np.median(finite_ranks)) if finite_ranks.size > 0 else np.nan,

    }



    for k in RANK_METRIC_CUTOFFS:

        kk = min(int(k), int(topk_per_source))

        hits_k = int(np.sum(finite_ranks <= kk)) if finite_ranks.size > 0 else 0

        out[f"{prefix}_hits_at_{k}"] = hits_k

        out[f"{prefix}_hits_at_{k}_rate"] = hits_k / total_true_edges



    return out





class PathJODIE(nn.Module):

    def __init__(self, embedding_dim: int, num_users: int, num_items_with_none: int, num_features: int = 1):

        super().__init__()

        args = SimpleNamespace(model="jodie", embedding_dim=int(embedding_dim))

        self.original = OriginalJODIE(

            args=args,

            num_features=int(num_features),

            num_users=int(num_users),

            num_items=int(num_items_with_none),

        )

        self.embedding_dim = int(embedding_dim)

        self.num_users = int(num_users)

        self.num_items = int(num_items_with_none)

        self.num_features = int(num_features)





        self.original.initial_user_embedding = nn.Parameter(

            F.normalize(torch.rand(self.embedding_dim), dim=0)

        )

        self.original.initial_item_embedding = nn.Parameter(

            F.normalize(torch.rand(self.embedding_dim), dim=0)

        )



    @property

    def initial_user_embedding(self):

        return self.original.initial_user_embedding



    @property

    def initial_item_embedding(self):

        return self.original.initial_item_embedding



    def project_user(self, user_dyn: torch.Tensor, user_timediff: torch.Tensor) -> torch.Tensor:

        dummy_item = user_dyn

        dummy_features = torch.zeros(

            (user_dyn.shape[0], self.num_features),

            dtype=user_dyn.dtype,

            device=user_dyn.device,

        )

        return self.original.forward(

            user_dyn,

            dummy_item,

            timediffs=user_timediff,

            features=dummy_features,

            select="project",

        )



    def update_user(self, user_dyn: torch.Tensor, item_dyn: torch.Tensor, user_timediff: torch.Tensor, features: torch.Tensor) -> torch.Tensor:

        return self.original.forward(

            user_dyn,

            item_dyn,

            timediffs=user_timediff,

            features=features,

            select="user_update",

        )



    def update_item(self, user_dyn: torch.Tensor, item_dyn: torch.Tensor, item_timediff: torch.Tensor, features: torch.Tensor) -> torch.Tensor:

        return self.original.forward(

            user_dyn,

            item_dyn,

            timediffs=item_timediff,

            features=features,

            select="item_update",

        )



    def predict_item_embedding(self, x: torch.Tensor) -> torch.Tensor:

        return self.original.predict_item_embedding(x)





def _build_jodie_user_item_mapping(*edge_lists):

    vertices = sorted(

        {

            int(x)

            for edge_list in edge_lists

            for (u, v, _, _) in edge_list

            for x in (u, v)

        }

    )



    user_to_jodie = {x: idx for idx, x in enumerate(vertices)}

    item_to_jodie = {x: idx for idx, x in enumerate(vertices)}



    jodie_to_user = {idx: x for x, idx in user_to_jodie.items()}

    jodie_to_item = {idx: x for x, idx in item_to_jodie.items()}



    return user_to_jodie, item_to_jodie, jodie_to_user, jodie_to_item



def _scale_like_original_jodie(values):

    arr = np.asarray(values, dtype=np.float32) + 1.0

    if arr.size == 0:

        return arr, 0.0, 1.0



    mean = float(arr.mean())

    std = float(arr.std())

    if std < 1e-12 or not np.isfinite(std):

        std = 1.0



    return ((arr - mean) / std).astype(np.float32), mean, std





def _jodie_scale_timediff(delta, mean, std):

    x = max(0.0, float(delta)) + 1.0

    std = float(std) if abs(float(std)) > 1e-12 else 1.0

    return (x - float(mean)) / std





def _build_original_jodie_sequences(edges_train, user_to_jodie, item_to_jodie, none_item_id):

    rows = []

    for u, v, t, _ in sorted(edges_train, key=lambda e: (e[2], e[0], e[1])):

        u = int(u)

        v = int(v)

        if u in user_to_jodie and v in item_to_jodie:

            rows.append((u, v, float(t)))



    if not rows:

        raise ValueError("No valid JODIE training rows after user/item mapping.")



    time_origin = float(rows[0][2])



    user_sequence_id = []

    item_sequence_id = []

    timestamp_sequence = []

    feature_sequence = []

    raw_user_timediffs = []

    raw_item_timediffs = []

    user_previous_itemid_sequence = []



    user_current_timestamp = defaultdict(float)

    item_current_timestamp = defaultdict(float)

    user_latest_itemid = defaultdict(lambda: none_item_id)



    for u_orig, v_orig, raw_t in rows:

        uid = user_to_jodie[u_orig]

        iid = item_to_jodie[v_orig]

        rel_t = float(raw_t) - time_origin



        user_sequence_id.append(uid)

        item_sequence_id.append(iid)

        timestamp_sequence.append(rel_t)

        feature_sequence.append([0.0])



        raw_user_timediffs.append(rel_t - user_current_timestamp[u_orig])

        raw_item_timediffs.append(rel_t - item_current_timestamp[v_orig])



        user_current_timestamp[u_orig] = rel_t

        item_current_timestamp[v_orig] = rel_t



        user_previous_itemid_sequence.append(user_latest_itemid[u_orig])

        user_latest_itemid[u_orig] = iid



    user_timediffs_scaled, user_td_mean, user_td_std = _scale_like_original_jodie(raw_user_timediffs)

    item_timediffs_scaled, item_td_mean, item_td_std = _scale_like_original_jodie(raw_item_timediffs)



    return {

        "user_sequence_id": np.asarray(user_sequence_id, dtype=np.int64),

        "item_sequence_id": np.asarray(item_sequence_id, dtype=np.int64),

        "timestamp_sequence": np.asarray(timestamp_sequence, dtype=np.float32),

        "feature_sequence": np.asarray(feature_sequence, dtype=np.float32),

        "user_timediffs_sequence": np.asarray(user_timediffs_scaled, dtype=np.float32),

        "item_timediffs_sequence": np.asarray(item_timediffs_scaled, dtype=np.float32),

        "user_previous_itemid_sequence": np.asarray(user_previous_itemid_sequence, dtype=np.int64),

        "time_origin": time_origin,

        "user_time_mean": user_td_mean,

        "user_time_std": user_td_std,

        "item_time_mean": item_td_mean,

        "item_time_std": item_td_std,

    }





def _new_tbatch_containers():

    return {

        "user": defaultdict(list),

        "item": defaultdict(list),

        "interactionids": defaultdict(list),

        "feature": defaultdict(list),

        "user_timediffs": defaultdict(list),

        "item_timediffs": defaultdict(list),

        "previous_item": defaultdict(list),

    }





def _snapshot_tbatch_containers(cur):

    return {

        key: {int(k): list(v) for k, v in value.items()}

        for key, value in cur.items()

    }





def _build_original_jodie_tbatch_groups(seq, flush_final=False):

    user_sequence_id = seq["user_sequence_id"]

    item_sequence_id = seq["item_sequence_id"]

    timestamp_sequence = seq["timestamp_sequence"]

    feature_sequence = seq["feature_sequence"]

    user_timediffs_sequence = seq["user_timediffs_sequence"]

    item_timediffs_sequence = seq["item_timediffs_sequence"]

    user_previous_itemid_sequence = seq["user_previous_itemid_sequence"]



    num_interactions = len(user_sequence_id)

    if num_interactions == 0:

        return []



    timespan = float(timestamp_sequence[-1] - timestamp_sequence[0])

    tbatch_timespan = timespan / 500.0 if timespan > 0 else 1.0



    groups = []

    cur = _new_tbatch_containers()

    tbatchid_user = defaultdict(lambda: -1)

    tbatchid_item = defaultdict(lambda: -1)

    tbatch_start_time = None



    for j in range(num_interactions):

        userid = int(user_sequence_id[j])

        itemid = int(item_sequence_id[j])

        timestamp = float(timestamp_sequence[j])



        tbatch_to_insert = max(tbatchid_user[userid], tbatchid_item[itemid]) + 1

        tbatchid_user[userid] = tbatch_to_insert

        tbatchid_item[itemid] = tbatch_to_insert



        cur["user"][tbatch_to_insert].append(userid)

        cur["item"][tbatch_to_insert].append(itemid)

        cur["interactionids"][tbatch_to_insert].append(j)

        cur["feature"][tbatch_to_insert].append(feature_sequence[j].tolist())

        cur["user_timediffs"][tbatch_to_insert].append(float(user_timediffs_sequence[j]))

        cur["item_timediffs"][tbatch_to_insert].append(float(item_timediffs_sequence[j]))

        cur["previous_item"][tbatch_to_insert].append(int(user_previous_itemid_sequence[j]))



        if tbatch_start_time is None:

            tbatch_start_time = timestamp



        if timestamp - tbatch_start_time > tbatch_timespan:

            groups.append(_snapshot_tbatch_containers(cur))

            tbatch_start_time = timestamp

            cur = _new_tbatch_containers()

            tbatchid_user = defaultdict(lambda: -1)

            tbatchid_item = defaultdict(lambda: -1)



    if flush_final and len(cur["user"]) > 0:

        groups.append(_snapshot_tbatch_containers(cur))



    return groups





def _jodie_initial_state(model: PathJODIE, device):

    user_dyn = F.normalize(model.initial_user_embedding, dim=0).repeat(model.num_users, 1).detach().clone().to(device)

    item_dyn = F.normalize(model.initial_item_embedding, dim=0).repeat(model.num_items, 1).detach().clone().to(device)

    user_static = torch.eye(model.num_users, device=device)

    item_static = torch.eye(model.num_items, device=device)

    return user_dyn, item_dyn, user_static, item_static





def _train_jodie_model(edges_train, edges_prefix_for_nodes, seed=42, device=None, n_epochs=5, embedding_dim=128, lr=1e-3, weight_decay=1e-5):

    if seed is None:

        seed = random.randint(0, 2**31 - 1)



    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)



    if device is None:

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    user_to_jodie, item_to_jodie, jodie_to_user, jodie_to_item = _build_jodie_user_item_mapping(edges_prefix_for_nodes)



    num_users = len(user_to_jodie)

    num_items = len(item_to_jodie)

    none_item_id = num_items

    num_items_with_none = num_items + 1

    num_features = 1



    seq = _build_original_jodie_sequences(

        edges_train=edges_train,

        user_to_jodie=user_to_jodie,

        item_to_jodie=item_to_jodie,

        none_item_id=none_item_id,

    )



    tbatch_groups = _build_original_jodie_tbatch_groups(seq, flush_final=JODIE_FLUSH_FINAL_TBATCH)



    if not tbatch_groups:

        raise ValueError(

            "Original JODIE T-batch construction produced no training groups. "

            "Set JODIE_FLUSH_FINAL_TBATCH=True for a practical fallback."

        )



    model = PathJODIE(

        embedding_dim=embedding_dim,

        num_users=num_users,

        num_items_with_none=num_items_with_none,

        num_features=num_features,

    ).to(device)



    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    mse_loss = nn.MSELoss()



    user_embedding_static = torch.eye(num_users, device=device)

    item_embedding_static = torch.eye(num_items_with_none, device=device)



    print(

        f"*** Training JODIE with original T-batch scheme: "

        f"epochs={n_epochs}, tbatch_windows={len(tbatch_groups)}, "

        f"num_users={num_users}, num_items={num_items_with_none} ***"

    )



    for ep in range(n_epochs):

        epoch_start_time = time_lib.time()



        user_embeddings = model.initial_user_embedding.repeat(num_users, 1)

        item_embeddings = model.initial_item_embedding.repeat(num_items_with_none, 1)



        user_embeddings_timeseries = torch.empty((len(seq["user_sequence_id"]), embedding_dim), device=device)

        item_embeddings_timeseries = torch.empty((len(seq["item_sequence_id"]), embedding_dim), device=device)



        optimizer.zero_grad()

        total_loss = 0.0

        total_interaction_count = 0



        for group in tbatch_groups:

            loss = None

            for tbatch_id in sorted(group["user"].keys()):

                user_ids_list = group["user"][tbatch_id]

                item_ids_list = group["item"][tbatch_id]

                if len(user_ids_list) == 0:

                    continue



                total_interaction_count += len(user_ids_list)



                tbatch_userids = torch.LongTensor(user_ids_list).to(device)

                tbatch_itemids = torch.LongTensor(item_ids_list).to(device)

                tbatch_interactionids = torch.LongTensor(group["interactionids"][tbatch_id]).to(device)

                feature_tensor = torch.Tensor(group["feature"][tbatch_id]).to(device)

                user_timediffs_tensor = torch.Tensor(group["user_timediffs"][tbatch_id]).to(device).unsqueeze(1)

                item_timediffs_tensor = torch.Tensor(group["item_timediffs"][tbatch_id]).to(device).unsqueeze(1)

                tbatch_itemids_previous = torch.LongTensor(group["previous_item"][tbatch_id]).to(device)



                item_embedding_previous = item_embeddings[tbatch_itemids_previous, :]

                user_embedding_input = user_embeddings[tbatch_userids, :]



                user_projected_embedding = model.original.forward(

                    user_embedding_input,

                    item_embedding_previous,

                    timediffs=user_timediffs_tensor,

                    features=feature_tensor,

                    select="project",

                )



                user_item_embedding = torch.cat(

                    [

                        user_projected_embedding,

                        item_embedding_previous,

                        item_embedding_static[tbatch_itemids_previous, :],

                        user_embedding_static[tbatch_userids, :],

                    ],

                    dim=1,

                )



                predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                item_embedding_input = item_embeddings[tbatch_itemids, :]



                prediction_target = torch.cat(

                    [item_embedding_input, item_embedding_static[tbatch_itemids, :]],

                    dim=1,

                ).detach()



                batch_loss = mse_loss(predicted_item_embedding, prediction_target)



                user_embedding_output = model.original.forward(

                    user_embedding_input,

                    item_embedding_input,

                    timediffs=user_timediffs_tensor,

                    features=feature_tensor,

                    select="user_update",

                )



                item_embedding_output = model.original.forward(

                    user_embedding_input,

                    item_embedding_input,

                    timediffs=item_timediffs_tensor,

                    features=feature_tensor,

                    select="item_update",

                )



                batch_loss = batch_loss + mse_loss(item_embedding_output, item_embedding_input.detach())

                batch_loss = batch_loss + mse_loss(user_embedding_output, user_embedding_input.detach())



                loss = batch_loss if loss is None else loss + batch_loss



                next_item_embeddings = item_embeddings.clone()

                next_user_embeddings = user_embeddings.clone()

                next_item_embeddings[tbatch_itemids, :] = item_embedding_output

                next_user_embeddings[tbatch_userids, :] = user_embedding_output

                item_embeddings = next_item_embeddings

                user_embeddings = next_user_embeddings



                user_embeddings_timeseries[tbatch_interactionids, :] = user_embedding_output

                item_embeddings_timeseries[tbatch_interactionids, :] = item_embedding_output



            if loss is not None:

                total_loss += float(loss.detach().cpu().item())

                loss.backward()

                optimizer.step()

                optimizer.zero_grad()



                item_embeddings = item_embeddings.detach()

                user_embeddings = user_embeddings.detach()

                item_embeddings_timeseries = item_embeddings_timeseries.detach()

                user_embeddings_timeseries = user_embeddings_timeseries.detach()



        elapsed_sec = (time_lib.time() - epoch_start_time)

        print(

            f"[JODIE-original] epoch={ep + 1}/{n_epochs} "

            f"total_loss={total_loss:.6f} "

            f"interactions={total_interaction_count} "

            f"seconds={elapsed_sec:.3f}"

        )



    return {

        "mode": None,

        "model": model,

        "user_to_jodie": user_to_jodie,

        "item_to_jodie": item_to_jodie,

        "jodie_to_user": jodie_to_user,

        "jodie_to_item": jodie_to_item,

        "none_item_id": none_item_id,

        "n_jodie_users": num_users,

        "n_jodie_items": num_items,

        "device": device,

        "time_origin": seq["time_origin"],

        "user_time_mean": seq["user_time_mean"],

        "user_time_std": seq["user_time_std"],

        "item_time_mean": seq["item_time_mean"],

        "item_time_std": seq["item_time_std"],

        "time_mean": seq["user_time_mean"],

        "time_std": seq["user_time_std"],

    }





@torch.no_grad()

def _jodie_warmup_state(oracle, edges_prefix):

    model: PathJODIE = oracle["model"]

    device = oracle["device"]

    user_to_jodie = oracle["user_to_jodie"]

    item_to_jodie = oracle["item_to_jodie"]

    none_item_id = oracle["none_item_id"]



    model.eval()

    user_dyn, item_dyn, user_static, item_static = _jodie_initial_state(model, device)



    last_user_time = defaultdict(float)

    last_item_time = defaultdict(float)

    prev_item_for_user = defaultdict(lambda: none_item_id)

    feature = torch.zeros((1, 1), dtype=torch.float32, device=device)

    time_origin = float(oracle.get("time_origin", 0.0))



    for u_orig, v_orig, t, _ in sorted(edges_prefix, key=lambda e: e[2]):

        u_orig = int(u_orig)

        v_orig = int(v_orig)

        if u_orig not in user_to_jodie or v_orig not in item_to_jodie:

            continue



        uid = user_to_jodie[u_orig]

        iid = item_to_jodie[v_orig]

        rel_t = float(t) - time_origin



        user_td = _jodie_scale_timediff(

            rel_t - float(last_user_time[u_orig]),

            oracle["user_time_mean"],

            oracle["user_time_std"],

        )

        item_td = _jodie_scale_timediff(

            rel_t - float(last_item_time[v_orig]),

            oracle["item_time_mean"],

            oracle["item_time_std"],

        )



        user_td_t = torch.tensor([[user_td]], dtype=torch.float32, device=device)

        item_td_t = torch.tensor([[item_td]], dtype=torch.float32, device=device)



        u_in = user_dyn[uid:uid + 1]

        i_in = item_dyn[iid:iid + 1]



        u_out = model.update_user(u_in, i_in, user_td_t, feature)

        i_out = model.update_item(u_in, i_in, item_td_t, feature)



        user_dyn[uid] = u_out.squeeze(0)

        item_dyn[iid] = i_out.squeeze(0)



        last_user_time[u_orig] = rel_t

        last_item_time[v_orig] = rel_t

        prev_item_for_user[uid] = iid



    oracle["user_dyn"] = user_dyn.detach()

    oracle["item_dyn"] = item_dyn.detach()

    oracle["user_static"] = user_static.detach()

    oracle["item_static"] = item_static.detach()

    oracle["last_user_time"] = dict(last_user_time)

    oracle["last_item_time"] = dict(last_item_time)

    oracle["prev_item_for_user"] = dict(prev_item_for_user)

    return oracle





@torch.no_grad()

def _jodie_update_state_one_edge_inplace(oracle, u_orig, v_orig, t_cur):

    model: PathJODIE = oracle["model"]

    user_to_jodie = oracle["user_to_jodie"]

    item_to_jodie = oracle["item_to_jodie"]

    device = oracle["device"]



    u_orig = int(u_orig)

    v_orig = int(v_orig)

    t_cur = float(t_cur)



    if u_orig not in user_to_jodie or v_orig not in item_to_jodie:

        return



    uid = user_to_jodie[u_orig]

    iid = item_to_jodie[v_orig]



    last_user_time = oracle.setdefault("last_user_time", {})

    last_item_time = oracle.setdefault("last_item_time", {})

    prev_item_for_user = oracle.setdefault("prev_item_for_user", {})



    time_origin = float(oracle.get("time_origin", 0.0))

    rel_t = float(t_cur) - time_origin



    user_td = _jodie_scale_timediff(

        rel_t - float(last_user_time.get(u_orig, 0.0)),

        oracle["user_time_mean"],

        oracle["user_time_std"],

    )

    item_td = _jodie_scale_timediff(

        rel_t - float(last_item_time.get(v_orig, 0.0)),

        oracle["item_time_mean"],

        oracle["item_time_std"],

    )



    user_td_t = torch.tensor([[user_td]], dtype=torch.float32, device=device)

    item_td_t = torch.tensor([[item_td]], dtype=torch.float32, device=device)

    feature = torch.zeros((1, 1), dtype=torch.float32, device=device)



    u_in = oracle["user_dyn"][uid:uid + 1]

    i_in = oracle["item_dyn"][iid:iid + 1]



    u_out = model.update_user(u_in, i_in, user_td_t, feature)

    i_out = model.update_item(u_in, i_in, item_td_t, feature)



    oracle["user_dyn"][uid] = u_out.squeeze(0).detach()

    oracle["item_dyn"][iid] = i_out.squeeze(0).detach()



    last_user_time[u_orig] = rel_t

    last_item_time[v_orig] = rel_t

    prev_item_for_user[uid] = iid





def _clone_jodie_runtime_oracle(oracle):

    out = dict(oracle)

    for key in ["user_dyn", "item_dyn", "user_static", "item_static"]:

        if key in oracle and torch.is_tensor(oracle[key]):

            out[key] = oracle[key].detach().clone()

    for key in ["last_user_time", "last_item_time", "prev_item_for_user"]:

        if key in oracle:

            out[key] = dict(oracle[key])

    out["model"] = oracle["model"]

    return out





def _jodie_predicted_item_embedding_for_source(oracle, u_orig, t_cur):

    model: PathJODIE = oracle["model"]

    user_to_jodie = oracle["user_to_jodie"]

    device = oracle["device"]



    u_orig = int(u_orig)

    if u_orig not in user_to_jodie:

        return None



    uid = user_to_jodie[u_orig]

    none_item_id = oracle["none_item_id"]

    prev_iid = oracle.get("prev_item_for_user", {}).get(uid, none_item_id)

    time_origin = float(oracle.get("time_origin", 0.0))

    rel_t = float(t_cur) - time_origin

    last_time = oracle.get("last_user_time", {}).get(u_orig, 0.0)



    td = _jodie_scale_timediff(

        rel_t - float(last_time),

        oracle["user_time_mean"],

        oracle["user_time_std"],

    )

    td_t = torch.tensor([[td]], dtype=torch.float32, device=device)



    user_dyn = oracle["user_dyn"]

    item_dyn = oracle["item_dyn"]

    user_static = oracle["user_static"]

    item_static = oracle["item_static"]



    u_in = user_dyn[uid:uid + 1]

    prev_i = item_dyn[prev_iid:prev_iid + 1]

    u_proj = model.project_user(u_in, td_t)

    pred_input = torch.cat(

        [

            u_proj,

            prev_i,

            item_static[prev_iid:prev_iid + 1],

            user_static[uid:uid + 1],

        ],

        dim=1,

    )

    return model.predict_item_embedding(pred_input)





def _jodie_filter_candidate_pairs(oracle, candidate_pair_to_best_len, t_cur, topk_per_source=50):

    if not candidate_pair_to_best_len:

        return []



    user_to_jodie = oracle["user_to_jodie"]

    item_to_jodie = oracle["item_to_jodie"]



    valid_pairs = []

    sources = []

    source_to_row = {}



    for (u, v), new_len in candidate_pair_to_best_len.items():

        u = int(u)

        v = int(v)

        if u not in user_to_jodie or v not in item_to_jodie:

            continue

        if u not in source_to_row:

            source_to_row[u] = len(sources)

            sources.append(u)

        valid_pairs.append((u, v, int(new_len), source_to_row[u]))



    if not valid_pairs:

        return []



    model: PathJODIE = oracle["model"]

    device = oracle["device"]

    model.eval()



    with torch.no_grad():

        time_origin = float(oracle.get("time_origin", 0.0))

        rel_t = float(t_cur) - time_origin

        last_user_time = oracle.get("last_user_time", {})

        prev_item_for_user = oracle.get("prev_item_for_user", {})

        none_item_id = int(oracle["none_item_id"])



        source_uids = [int(user_to_jodie[u]) for u in sources]

        source_prev_iids = [

            int(prev_item_for_user.get(int(user_to_jodie[u]), none_item_id))

            for u in sources

        ]

        source_tds = [

            _jodie_scale_timediff(

                rel_t - float(last_user_time.get(u, 0.0)),

                oracle["user_time_mean"],

                oracle["user_time_std"],

            )

            for u in sources

        ]



        uid_t = torch.as_tensor(source_uids, dtype=torch.long, device=device)

        prev_iid_t = torch.as_tensor(source_prev_iids, dtype=torch.long, device=device)

        td_t = torch.as_tensor(source_tds, dtype=torch.float32, device=device).view(-1, 1)



        user_dyn = oracle["user_dyn"]

        item_dyn = oracle["item_dyn"]

        user_static = oracle["user_static"]

        item_static = oracle["item_static"]



        u_in = user_dyn[uid_t]

        prev_i = item_dyn[prev_iid_t]

        u_proj = model.project_user(u_in, td_t)

        pred_input = torch.cat(

            [

                u_proj,

                prev_i,

                item_static[prev_iid_t],

                user_static[uid_t],

            ],

            dim=1,

        )

        pred_by_source = model.predict_item_embedding(pred_input)



        pair_source_rows = torch.as_tensor(

            [src_row for _u, _v, _new_len, src_row in valid_pairs],

            dtype=torch.long,

            device=device,

        )

        pair_iids = torch.as_tensor(

            [int(item_to_jodie[v]) for _u, v, _new_len, _src_row in valid_pairs],

            dtype=torch.long,

            device=device,

        )



        target_emb = torch.cat([item_dyn[pair_iids], item_static[pair_iids]], dim=1)

        pred_for_pair = pred_by_source[pair_source_rows]

        distances = torch.norm(target_emb - pred_for_pair, p=2, dim=1).detach().cpu().numpy()



    grouped_pair_indices = defaultdict(list)

    for idx, (u, _v, _new_len, _src_row) in enumerate(valid_pairs):

        grouped_pair_indices[int(u)].append(idx)



    out = []

    use_topk = topk_per_source is not None and int(topk_per_source) > 0

    topk = int(topk_per_source) if use_topk else None



    for u in sorted(grouped_pair_indices.keys()):

        indices = grouped_pair_indices[u]



        if use_topk and len(indices) > topk:

            local_distances = distances[indices]

            selected_local = np.argpartition(local_distances, topk - 1)[:topk]

            selected_indices = [indices[int(i)] for i in selected_local]

        else:

            selected_indices = indices





        selected_indices.sort(key=lambda idx: (float(distances[idx]), int(valid_pairs[idx][1])))



        for rank, idx in enumerate(selected_indices, start=1):

            u_i, v_i, new_len_i, _src_row = valid_pairs[idx]

            out.append(

                {

                    "u": int(u_i),

                    "v": int(v_i),

                    "new_len": int(new_len_i),

                    "distance": float(distances[idx]),

                    "rank": int(rank),

                }

            )



    return out





def _jodie_distance_for_edge(oracle, u, v, t_cur):

    user_to_jodie = oracle["user_to_jodie"]

    item_to_jodie = oracle["item_to_jodie"]

    u = int(u)

    v = int(v)

    if u not in user_to_jodie or v not in item_to_jodie:

        return None



    model: PathJODIE = oracle["model"]

    model.eval()

    with torch.no_grad():

        pred = _jodie_predicted_item_embedding_for_source(oracle, u, t_cur)

        if pred is None:

            return None

        iid = item_to_jodie[v]

        target = torch.cat([oracle["item_dyn"][iid:iid + 1], oracle["item_static"][iid:iid + 1]], dim=1)

        return float(torch.norm(target - pred, p=2, dim=1).detach().cpu().item())





def _fit_jodie_tau_from_validation(oracle, edges_val):

    distances = []

    for u, v, t, _ in edges_val:

        d = _jodie_distance_for_edge(oracle, u, v, t)

        if d is not None and np.isfinite(d):

            distances.append(d)

    if not distances:

        return 1.0



    tau = float(np.mean(distances))

    if tau <= 1e-12 or not np.isfinite(tau):

        tau = 1.0

    return tau





def _jodie_probability_from_distance(distance, tau):

    tau = max(float(tau), 1e-12)

    return float(np.exp(-float(distance) / tau))





def _score_jodie_rows(oracle, rows):

    tau = float(oracle.get("tau", 1.0))

    out = []

    for r in rows:

        if oracle["mode"] == "jodie_distance" or JODIE_RR_SCORE_MODE == "exp_distance":

            p = _jodie_probability_from_distance(float(r["distance"]), tau)

        else:

            p = 1.0 / max(1, int(r["rank"]))

        out.append((int(r["u"]), int(r["v"]), int(r["new_len"]), float(p)))

    return out





def _score_jodie_candidates_batch(oracle, candidate_pair_to_best_len, t_cur, topk_per_source=50):

    rows = _jodie_filter_candidate_pairs(

        oracle=oracle,

        candidate_pair_to_best_len=candidate_pair_to_best_len,

        t_cur=t_cur,

        topk_per_source=topk_per_source,

    )

    if not rows:

        return []

    return _score_jodie_rows(oracle, rows)





def _evaluate_jodie_oracle_on_window(

    oracle,

    edges_future,

    vertex_list,

    future_times_eval,

    lp_edge_thresholds,

    topk_per_source=50,

    seed=42,

    label_suffix="eval",

):

    eval_edges = sorted(edges_future, key=lambda e: (e[2], e[0], e[1]))



    y_true = []

    y_score = []

    candidate_ranks_for_true_edges = []

    score_ranks_for_true_edges = []

    true_edge_scores = []



    all_targets = sorted(set(int(x) for x in vertex_list if int(x) in oracle["item_to_jodie"]))



    for u, v_true, t, _ in eval_edges:

        u = int(u)

        v_true = int(v_true)

        t = int(t)



        if u not in oracle["user_to_jodie"] or v_true not in oracle["item_to_jodie"]:

            y_true.append(1)

            y_score.append(MISSING_TRUE_EDGE_SCORE)

            true_edge_scores.append(MISSING_TRUE_EDGE_SCORE)

            candidate_ranks_for_true_edges.append(math.inf)

            score_ranks_for_true_edges.append(math.inf)

            if JODIE_ONLINE_EVAL_UPDATE_TRUE_EDGES:

                _jodie_update_state_one_edge_inplace(oracle, u, v_true, t)

            continue



        candidate_pair_to_best_len = {

            (u, int(v)): 1

            for v in all_targets

            if int(v) != u

        }



        rows = _jodie_filter_candidate_pairs(

            oracle=oracle,

            candidate_pair_to_best_len=candidate_pair_to_best_len,

            t_cur=t,

            topk_per_source=topk_per_source,

        )





        scored = _score_jodie_rows(oracle, rows)



        score_by_v = {int(v): float(p) for _, v, _, p in scored}

        candidate_rank_by_v = {int(r["v"]): int(r["rank"]) for r in rows}

        scored_sorted = sorted(scored, key=lambda x: (-float(x[3]), int(x[1])))

        score_rank_by_v = {

            int(v): rank

            for rank, (_, v, _, _) in enumerate(scored_sorted, start=1)

        }



        seen_v = set(score_by_v.keys())



        for _, v, _, p_hat in scored:

            v = int(v)

            y_true.append(1 if v == v_true else 0)

            y_score.append(float(p_hat))



        if v_true in seen_v:

            p_true = float(score_by_v[v_true])

            true_edge_scores.append(p_true)

            candidate_ranks_for_true_edges.append(float(candidate_rank_by_v.get(v_true, math.inf)))

            score_ranks_for_true_edges.append(float(score_rank_by_v.get(v_true, math.inf)))

        else:

            y_true.append(1)

            y_score.append(MISSING_TRUE_EDGE_SCORE)

            true_edge_scores.append(MISSING_TRUE_EDGE_SCORE)

            candidate_ranks_for_true_edges.append(math.inf)

            score_ranks_for_true_edges.append(math.inf)



        if JODIE_ONLINE_EVAL_UPDATE_TRUE_EDGES:

            _jodie_update_state_one_edge_inplace(oracle, u, v_true, t)



    if not y_true:

        raise ValueError("JODIE evaluation produced no examples.")



    total_true_edges = len(true_edge_scores)

    candidate_metrics = _rank_metric_dict(

        prefix="candidate_rank",

        ranks=candidate_ranks_for_true_edges,

        total_true_edges=total_true_edges,

        topk_per_source=topk_per_source,

    )

    score_metrics = _rank_metric_dict(

        prefix="score_rank",

        ranks=score_ranks_for_true_edges,

        total_true_edges=total_true_edges,

        topk_per_source=topk_per_source,

    )



    candidate_metrics["candidate_rank_recall_at_10"] = candidate_metrics.get(

        "candidate_rank_hits_at_10_rate", np.nan

    )

    score_metrics["score_rank_recall_at_10"] = score_metrics.get(

        "score_rank_hits_at_10_rate", np.nan

    )



    best_threshold, table = _threshold_table_from_scores(

        y_true=y_true,

        y_score=y_score,

        lp_edge_thresholds=lp_edge_thresholds,

        label=f"{oracle.get('mode', 'jodie')} online full-pipeline {label_suffix}",

    )



    for key, value in candidate_metrics.items():

        table[key] = value

    for key, value in score_metrics.items():

        table[key] = value



    for idx, row in table.iterrows():

        thr = float(row["threshold"])

        accepted_hits = int(sum(1 for s in true_edge_scores if float(s) >= thr))

        table.at[idx, "accepted_hits"] = accepted_hits

        table.at[idx, "accepted_recall"] = accepted_hits / total_true_edges if total_true_edges > 0 else np.nan



    print(

        f"[{oracle.get('mode')} online {label_suffix} ranking metrics] "

        f"topk={topk_per_source} "

        f"thresholds={lp_edge_thresholds} "

        f"true_edges={total_true_edges} "

        f"candidate_topk_recall={candidate_metrics['candidate_rank_topk_recall']:.6f} "

        f"candidate_mrr={candidate_metrics['candidate_rank_mrr']:.6f} "

        f"candidate_hits@1={candidate_metrics['candidate_rank_hits_at_1_rate']:.6f} "

        f"candidate_hits@5={candidate_metrics['candidate_rank_hits_at_5_rate']:.6f} "

        f"candidate_recall@10={candidate_metrics['candidate_rank_recall_at_10']:.6f} "

        f"candidate_mean_hit_rank={candidate_metrics['candidate_rank_mean_hit_rank']:.6f} "

        f"score_mrr={score_metrics['score_rank_mrr']:.6f} "

        f"score_hits@1={score_metrics['score_rank_hits_at_1_rate']:.6f} "

        f"score_hits@5={score_metrics['score_rank_hits_at_5_rate']:.6f} "

        f"score_recall@10={score_metrics['score_rank_recall_at_10']:.6f}"

    )



    return best_threshold, table





def _train_jodie_oracle(

    logic,

    edges_80,

    edges_80_90,

    edges_90,

    vertex_list_80,

    future_times_80_90,

    seed=42,

    device=None,

    jodie_epochs=5,

    jodie_embedding_dim=128,

    jodie_lr=1e-3,

    jodie_weight_decay=1e-5,

    jodie_topk_per_source=50,

):

    if logic not in {"jodie_rr", "jodie_distance"}:

        raise ValueError("This updated file supports LP_ORACLE_MODE='jodie_rr' or 'jodie_distance'.")



    start = time_lib.perf_counter()

    oracle = _train_jodie_model(

        edges_train=edges_80,

        edges_prefix_for_nodes=edges_90,

        seed=seed,

        device=device,

        n_epochs=jodie_epochs,

        embedding_dim=jodie_embedding_dim,

        lr=jodie_lr,

        weight_decay=jodie_weight_decay,

    )

    train_jodie_s = time_lib.perf_counter() - start



    oracle["mode"] = logic





    oracle["topk_per_source"] = jodie_topk_per_source





    oracle["path_topk_per_source"] = jodie_topk_per_source

    oracle["eval_topk_per_source"] = JODIE_EVAL_TOPK_PER_SOURCE



    _jodie_warmup_state(oracle, edges_80)

    oracle["tau"] = _fit_jodie_tau_from_validation(oracle, edges_80_90)



    if logic == "jodie_rr":

        print(

            "[jodie_rr] using original-JODIE item-embedding Euclidean ranking "

            "with reciprocal-rank edge score; no second-stage scorer is trained."

        )

    else:

        print("[jodie_distance] using original-JODIE Euclidean distance with exp(-distance/tau); no second-stage scorer is trained.")



    oracle["train_jodie_s"] = train_jodie_s

    oracle["train_second_stage_s"] = 0.0

    return oracle





def split_path_str(s: Any) -> List[int]:

    if pd.isna(s):

        return []

    text = str(s).strip()

    if not text or text.lower() in {"nan", "none", "null", "[]"}:

        return []

    return [int(float(x)) for x in text.split(",") if str(x).strip()]





def edge_pairs_from_nodes(nodes: Sequence[int]) -> List[Tuple[int, int]]:

    return [(int(nodes[i]), int(nodes[i + 1])) for i in range(len(nodes) - 1)]





def build_pair_time_index(edges: Iterable[Tuple[int, int, int, float]]) -> Dict[Tuple[int, int], List[int]]:

    out: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for u, v, t, _p in edges:

        out[(int(u), int(v))].append(int(t))

    for key in list(out.keys()):

        out[key] = sorted(set(out[key]))

    return out





def infer_true_temporal_edges(

    true_nodes: Sequence[int],

    pair_times: Dict[Tuple[int, int], List[int]],

    pivot_time: int,

    query_time: int,

) -> List[Tuple[int, int, Optional[int]]]:

    edges: List[Tuple[int, int, Optional[int]]] = []

    last_t = int(pivot_time)

    for u, v in edge_pairs_from_nodes(true_nodes):

        candidates = [t for t in pair_times.get((u, v), []) if last_t < int(t) <= int(query_time)]

        if not candidates:

            edges.append((int(u), int(v), None))

            last_t = int(query_time) + 1

            continue

        chosen_t = int(candidates[0])

        edges.append((int(u), int(v), chosen_t))

        last_t = chosen_t

    return edges





def choose_temporal_reconstruction(

    true_pairs: Sequence[Tuple[int, int]],

    accepted_edges: Sequence[Dict[str, Any]],

    pivot_time: int,

    query_time: int,

) -> Tuple[bool, List[Dict[str, Any]]]:

    by_pair: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)

    for edge in accepted_edges:

        t = int(edge["time"])

        if int(pivot_time) < t <= int(query_time):

            by_pair[(int(edge["u"]), int(edge["v"]))].append(edge)



    for key in by_pair:

        by_pair[key] = sorted(by_pair[key], key=lambda e: (int(e["time"]), -float(e.get("score", 0.0))))



    selected: List[Dict[str, Any]] = []

    last_t = int(pivot_time)

    for u, v in true_pairs:

        candidates = [e for e in by_pair.get((int(u), int(v)), []) if int(e["time"]) > last_t]

        if not candidates:

            return False, []

        chosen = candidates[0]

        selected.append(chosen)

        last_t = int(chosen["time"])



    return True, selected





def safe_product(values: Sequence[float]) -> float:

    prod = 1.0

    for value in values:

        prod *= float(value)

    return float(prod)





def compute_query_result(

    dataset: str,

    iteration: int,

    method: str,

    row: Any,

    pivot_time: int,

    pair_times: Dict[Tuple[int, int], List[int]],

    accepted_edges: Sequence[Dict[str, Any]],

    selected_tuple: Tuple[Any, ...],

) -> Dict[str, Any]:

    src = int(row.source)

    dst = int(row.destination)

    query_time = int(row.time)

    true_nodes = split_path_str(row.future_Path)

    true_pairs = edge_pairs_from_nodes(true_nodes)

    true_temporal_edges = infer_true_temporal_edges(true_nodes, pair_times, pivot_time, query_time)

    true_pair_multiset = Counter(true_pairs)

    true_pair_set = set(true_pairs)

    true_len = max(0, len(true_nodes) - 1)



    accepted_valid = [

        e for e in accepted_edges

        if int(pivot_time) < int(e["time"]) <= query_time

    ]

    accepted_unique_keys = {

        (int(e["u"]), int(e["v"]), int(e["time"]))

        for e in accepted_valid

    }



    accepted_true_hits = 0

    matched_true_edge_count = 0

    for pair, required_count in true_pair_multiset.items():

        count = sum(1 for e in accepted_valid if (int(e["u"]), int(e["v"])) == pair)

        if count > 0:

            matched_true_edge_count += min(required_count, count)

    accepted_true_hits = sum(1 for e in accepted_valid if (int(e["u"]), int(e["v"])) in true_pair_set)



    edge_recall = matched_true_edge_count / max(1, len(true_pairs)) if true_pairs else np.nan

    edge_precision = accepted_true_hits / max(1, len(accepted_valid)) if accepted_valid else 0.0



    temporal_path_recovered, true_reconstruction_edges = choose_temporal_reconstruction(

        true_pairs=true_pairs,

        accepted_edges=accepted_valid,

        pivot_time=pivot_time,

        query_time=query_time,

    )



    selected_d = selected_tuple[0] if len(selected_tuple) > 0 else math.inf

    selected_time = selected_tuple[1] if len(selected_tuple) > 1 else math.inf

    selected_nodes = list(selected_tuple[2]) if len(selected_tuple) > 2 and selected_tuple[2] else []

    selected_edges = list(selected_tuple[3]) if len(selected_tuple) > 3 and selected_tuple[3] else []

    selected_prob = float(selected_tuple[4]) if len(selected_tuple) > 4 else np.nan

    selected_exist = float(selected_tuple[5]) if len(selected_tuple) > 5 else np.nan



    has_future_prediction = bool(selected_nodes) and selected_time != math.inf and int(selected_time) > int(pivot_time)

    selected_len = max(0, len(selected_nodes) - 1) if selected_nodes else math.inf

    direct_shortcut_selected = int(has_future_prediction and selected_len == 1 and true_len > 1)

    shorter_than_true_selected = int(has_future_prediction and selected_len < true_len)

    prefix_fallback = int(not has_future_prediction)



    direct_shortcut_edges = [

        e for e in accepted_valid

        if int(e["v"]) == dst and int(e.get("new_len", 10**9)) < true_len

    ]

    direct_shortcut_accepted = int(len(direct_shortcut_edges) > 0)



    true_accepted_scores = [

        float(e.get("score", 0.0))

        for e in accepted_valid

        if (int(e["u"]), int(e["v"])) in true_pair_set

    ]

    shortcut_scores = [float(e.get("score", 0.0)) for e in direct_shortcut_edges]



    shortcut_edge_dominance = np.nan

    strong_shortcut_edge_dominance = np.nan

    if shortcut_scores and true_accepted_scores:

        max_shortcut = max(shortcut_scores)

        shortcut_edge_dominance = float(max_shortcut > min(true_accepted_scores))

        strong_shortcut_edge_dominance = float(max_shortcut > max(true_accepted_scores))



    shortcut_path_dominance = np.nan

    if temporal_path_recovered and direct_shortcut_edges:

        true_path_score = safe_product([float(e.get("score", 0.0)) for e in true_reconstruction_edges])

        best_shortcut_score = max(float(e.get("score", 0.0)) for e in direct_shortcut_edges)

        shortcut_path_dominance = float(best_shortcut_score > true_path_score)

    else:

        true_path_score = np.nan

        best_shortcut_score = max(shortcut_scores) if shortcut_scores else np.nan



    return {

        "dataset": dataset,

        "iteration": int(iteration),

        "method": method,

        "source": src,

        "destination": dst,

        "query_time": query_time,

        "pivot_time": int(pivot_time),

        "true_nodes": ",".join(map(str, true_nodes)),

        "true_len": int(true_len),

        "true_pairs": ";".join(f"{u}->{v}" for u, v in true_pairs),

        "true_temporal_edges_inferred": ";".join(

            f"{u}->{v}@{t if t is not None else 'NA'}" for u, v, t in true_temporal_edges

        ),

        "num_true_future_edges": int(len(true_pairs)),

        "num_accepted_edges": int(len(accepted_valid)),

        "num_unique_accepted_edges": int(len(accepted_unique_keys)),

        "temporal_edge_recall": float(edge_recall),

        "temporal_edge_precision": float(edge_precision),

        "temporal_path_recall": float(temporal_path_recovered),

        "true_path_recovered": int(temporal_path_recovered),

        "future_coverage": float(has_future_prediction),

        "prefix_fallback": int(prefix_fallback),

        "selected_nodes": ",".join(map(str, selected_nodes)),

        "selected_len": selected_len if selected_len != math.inf else np.nan,

        "selected_time": selected_time if selected_time != math.inf else np.nan,

        "selected_prob": selected_prob,

        "selected_exist": selected_exist,

        "direct_shortcut_accepted": int(direct_shortcut_accepted),

        "direct_shortcut_selected": int(direct_shortcut_selected),

        "shorter_than_true_selected": int(shorter_than_true_selected),

        "shortcut_edge_dominance": shortcut_edge_dominance,

        "strong_shortcut_edge_dominance": strong_shortcut_edge_dominance,

        "shortcut_path_dominance": shortcut_path_dominance,

        "true_path_score_product": true_path_score,

        "best_direct_shortcut_score": best_shortcut_score,

        "accepted_edges": ";".join(

            f"{int(e['u'])}->{int(e['v'])}@{int(e['time'])}|len={int(e.get('new_len', -1))}|p={float(e.get('score', 0.0)):.6g}"

            for e in accepted_valid

        ),

    }





def _method_slug(method: str) -> str:

    return str(method).replace(" ", "_").replace("/", "_").replace("-", "_")





def _path_nodes_to_string(nodes: Sequence[Any]) -> str:

    if not nodes:

        return ""

    return ",".join(str(int(x)) for x in nodes)





def _path_edges_to_string(edges: Sequence[Any]) -> str:

    if not edges:

        return ""

    out = []

    for edge in edges:

        if len(edge) >= 4:

            u, v, t, prob = edge[:4]

            out.append(f"{int(u)}->{int(v)}@{int(t)}|p={float(prob):.8g}")

        elif len(edge) >= 3:

            u, v, t = edge[:3]

            out.append(f"{int(u)}->{int(v)}@{int(t)}")

    return " ; ".join(out)





def _safe_numeric(value: Any) -> Any:

    try:

        if value == math.inf:

            return np.nan

        return value

    except Exception:

        return value





def _topk_prediction_rows_for_query(

    dataset: str,

    file_stem: str,

    iteration: int,

    method: str,

    row: Any,

    pivot_time: int,

    topk_tuples: Sequence[Tuple[Any, ...]],

    max_rank: int,

) -> List[Dict[str, Any]]:

    src = int(row.source)

    dst = int(row.destination)

    query_time = int(row.time)

    true_nodes = split_path_str(row.future_Path)

    true_path = _path_nodes_to_string(true_nodes)

    true_len = max(0, len(true_nodes) - 1)



    out: List[Dict[str, Any]] = []

    for rank, cand in enumerate(list(topk_tuples)[: int(max_rank)], start=1):

        d_val = cand[0] if len(cand) > 0 else math.inf

        t_form = cand[1] if len(cand) > 1 else math.inf

        path_nodes = list(cand[2]) if len(cand) > 2 and cand[2] else []

        path_edges = list(cand[3]) if len(cand) > 3 and cand[3] else []

        p_sp = float(cand[4]) if len(cand) > 4 else np.nan

        p_exist = float(cand[5]) if len(cand) > 5 else np.nan

        predicted_len = max(0, len(path_nodes) - 1) if path_nodes else np.nan

        future_answer = bool(path_nodes) and t_form != math.inf and int(t_form) > int(pivot_time)



        out.append({

            "dataset": dataset,

            "file_stem": file_stem,

            "iteration": int(iteration),

            "method": method,

            "source": src,

            "destination": dst,

            "pivot_time": int(pivot_time),

            "query_time": query_time,

            "rank": int(rank),

            "truePath": true_path,

            "true_len": int(true_len),

            "predictedPath": _path_nodes_to_string(path_nodes),

            "predictedEdges": _path_edges_to_string(path_edges),

            "predictedDistance": _safe_numeric(d_val),

            "predictionTime": _safe_numeric(t_form),

            "predicted_len": predicted_len,

            "shortestPathProba": p_sp,

            "pathProba": p_exist,

            "future_coverage": float(future_answer),

        })



    return out





def _accepted_edge_rows_for_query(

    dataset: str,

    file_stem: str,

    iteration: int,

    method: str,

    row: Any,

    pivot_time: int,

    accepted_edges: Sequence[Dict[str, Any]],

) -> List[Dict[str, Any]]:

    src = int(row.source)

    dst = int(row.destination)

    query_time = int(row.time)

    out: List[Dict[str, Any]] = []

    for idx, edge in enumerate(accepted_edges, start=1):

        edge_time = int(edge["time"])

        if edge_time <= int(pivot_time) or edge_time > query_time:

            continue

        out.append({

            "dataset": dataset,

            "file_stem": file_stem,

            "iteration": int(iteration),

            "method": method,

            "source": src,

            "destination": dst,

            "pivot_time": int(pivot_time),

            "query_time": query_time,

            "edge_order": int(idx),

            "u": int(edge["u"]),

            "v": int(edge["v"]),

            "time": edge_time,

            "score": float(edge.get("score", 0.0)),

            "new_len": int(edge.get("new_len", -1)),

            "lv_path_id": int(edge.get("lv_path_id", -1)),

        })

    return out





def _flatten_predictsp_timing(timing_info: Dict[str, Any], dst: int) -> Dict[str, Any]:

    if not isinstance(timing_info, dict):

        return {}

    out: Dict[str, Any] = {

        "predictsp_total_s": float(timing_info.get("predictsp_total_s", 0.0)),

        "predictsp_stream_update_s": float(timing_info.get("predictsp_stream_update_s", 0.0)),

    }

    by_dst = timing_info.get("by_dst", {})

    dst_info = by_dst.get(dst, by_dst.get(int(dst), {})) if isinstance(by_dst, dict) else {}

    if isinstance(dst_info, dict):

        for key, value in dst_info.items():

            out[key] = value

    return out





def _timing_row_for_step(

    dataset: str,

    file_stem: str,

    iteration: int,

    method: str,

    row: Any,

    pivot_time: int,

    t_cur: int,

    step_total_s: float,

    candidate_pair_to_best_len: Dict[Any, Any],

    cached_scored_candidate_occurrences: Sequence[Any],

    missing_candidate_occurrences_by_pair: Dict[Any, Any],

    new_scored_candidates: Sequence[Any],

    scored_candidates: Sequence[Any],

    accepted_before_policy: Sequence[Any],

    accepted_candidates: Sequence[Any],

    candidate_build_stats: Dict[str, Any],

    edge_stats: Dict[str, Any],

    candidate_count_dict: Dict[Any, Any],

    timing_info: Dict[str, Any],

) -> Dict[str, Any]:

    src = int(row.source)

    dst = int(row.destination)

    before_lb, after_lb = candidate_count_dict.get(dst, candidate_count_dict.get(int(dst), (np.nan, np.nan)))

    out: Dict[str, Any] = {

        "dataset": dataset,

        "file_stem": file_stem,

        "iteration": int(iteration),

        "method": method,

        "source": src,

        "destination": dst,

        "pivot_time": int(pivot_time),

        "query_time": int(row.time),

        "t_cur": int(t_cur),

        "pipeline_step_total_s": float(step_total_s),

        "num_candidate_pairs": int(len(candidate_pair_to_best_len)),

        "num_cached_scored_candidate_occurrences": int(len(cached_scored_candidate_occurrences)),

        "num_missing_candidate_pairs": int(len(missing_candidate_occurrences_by_pair)),

        "num_new_scored_candidates": int(len(new_scored_candidates)),

        "num_scored_candidate_occurrences": int(len(scored_candidates)),

        "num_accepted_before_policy": int(len(accepted_before_policy)),

        "num_accepted_candidates": int(len(accepted_candidates)),

        "predictsp_candidate_paths_before_lb": before_lb,

        "predictsp_candidate_paths_after_lb": after_lb,

    }

    if isinstance(candidate_build_stats, dict):

        out.update({f"candidate_{k}": v for k, v in candidate_build_stats.items()})

    if isinstance(edge_stats, dict):

        out.update(edge_stats)

    out.update(_flatten_predictsp_timing(timing_info, dst))

    return out





def _write_method_artifacts(

    cfg: PipelineConfig,

    data: Dict[str, Any],

    iteration: int,

    method: str,

    prediction_rows: Sequence[Dict[str, Any]],

    timing_rows: Sequence[Dict[str, Any]],

    accepted_edge_rows: Sequence[Dict[str, Any]],

) -> None:

    run_name = f"{data['file_stem']}_{int(iteration)}"

    method_slug = _method_slug(method)



    path_dir = cfg.output_dir / "path_prediction_results" / run_name

    timing_dir = cfg.output_dir / "pipeline_timing" / run_name

    edge_dir = cfg.output_dir / "edge_candidates" / run_name

    path_dir.mkdir(parents=True, exist_ok=True)

    timing_dir.mkdir(parents=True, exist_ok=True)

    edge_dir.mkdir(parents=True, exist_ok=True)



    pred_df = pd.DataFrame(prediction_rows)

    if not pred_df.empty:

        pred_df.to_csv(path_dir / f"{method_slug}_top{int(cfg.top_k_result_paths)}_predictions.tsv", sep="\t", index=False)

        rank1_df = pred_df[pred_df["rank"].astype(int) == 1].copy()

        rank1_df.to_csv(path_dir / f"{method_slug}_rank1_predictions.tsv", sep="\t", index=False)

    else:

        pd.DataFrame().to_csv(path_dir / f"{method_slug}_top{int(cfg.top_k_result_paths)}_predictions.tsv", sep="\t", index=False)



    pd.DataFrame(timing_rows).to_csv(timing_dir / f"{method_slug}_timing.tsv", sep="\t", index=False)

    pd.DataFrame(accepted_edge_rows).to_csv(edge_dir / f"{method_slug}_accepted_edges.tsv", sep="\t", index=False)



def summarize_results(detail_df: pd.DataFrame) -> pd.DataFrame:

    metrics = [

        "temporal_edge_recall",

        "temporal_edge_precision",

        "temporal_path_recall",

        "future_coverage",

        "prefix_fallback",

        "direct_shortcut_accepted",

        "direct_shortcut_selected",

        "shorter_than_true_selected",

        "shortcut_edge_dominance",

        "strong_shortcut_edge_dominance",

        "shortcut_path_dominance",

    ]



    rows = []

    for (dataset, method), group in detail_df.groupby(["dataset", "method"], sort=False):

        out = {

            "dataset": dataset,

            "method": method,

            "num_queries": int(len(group)),

        }

        for metric in metrics:

            vals = pd.to_numeric(group[metric], errors="coerce")

            out[metric] = float(vals.mean()) if vals.notna().any() else np.nan

        rows.append(out)



    summary = pd.DataFrame(rows)

    if summary.empty:

        return summary



    summary["_dataset_order"] = pd.Categorical(summary["dataset"], categories=DATASET_ORDER, ordered=True)

    summary["_method_order"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)

    summary = summary.sort_values(["_dataset_order", "_method_order"]).drop(columns=["_dataset_order", "_method_order"])

    return summary.reset_index(drop=True)





def make_latex_table(summary: pd.DataFrame) -> str:

    cols = [

        ("temporal_edge_recall", "Temp. edge rec."),

        ("temporal_edge_precision", "Temp. edge prec."),

        ("temporal_path_recall", "Temp. path rec."),

        ("future_coverage", "Future cov."),

        ("prefix_fallback", "Prefix fallback"),

        ("direct_shortcut_accepted", "Shortcut acc."),

        ("direct_shortcut_selected", "Shortcut sel."),

        ("shortcut_path_dominance", "Shortcut dom."),

    ]



    lines = []

    lines.append(r"\begin{table*}[t]")

    lines.append(r"\centering")

    lines.append(r"\caption{Post-oracle temporal path recovery and shortcut results.}")

    lines.append(r"\label{tab:oracle_temporal_results}")

    lines.append(r"\scriptsize")

    lines.append(r"\setlength{\tabcolsep}{3pt}")

    lines.append(r"\renewcommand{\arraystretch}{0.95}")

    lines.append(r"\resizebox{\textwidth}{!}{%")

    lines.append(r"\begin{tabular}{ll" + "c" * len(cols) + r"}")

    lines.append(r"\toprule")

    header = ["Dataset", "Oracle"] + [rf"\shortstack{ {label.replace(' ', '\\\\ ')}} " for _, label in cols]

    lines.append(" & ".join(header) + r" \\")

    lines.append(r"\midrule")



    for dataset in DATASET_ORDER:

        dsub = summary[summary["dataset"] == dataset]

        if dsub.empty:

            continue

        for method in METHOD_ORDER:

            row = dsub[dsub["method"] == method]

            if row.empty:

                continue

            values = [dataset, method]

            for metric, _label in cols:

                value = row.iloc[0][metric]

                values.append("--" if pd.isna(value) else f"{float(value):.3f}")

            lines.append(" & ".join(values) + r" \\")

        lines.append(r"\midrule")



    if lines[-1] == r"\midrule":

        lines[-1] = r"\bottomrule"

    else:

        lines.append(r"\bottomrule")

    lines.append(r"\end{tabular}%")

    lines.append(r"}")

    lines.append(r"\end{table*}")

    return "\n".join(lines)





def prepare_dataset(csv_path: Path, cfg: PipelineConfig) -> Optional[Dict[str, Any]]:

    file_stem = csv_path.stem

    df = _normalize_df(str(csv_path))



    missing_times = set(_load_missing_times(file_stem))

    if missing_times:

        df = df[~df["time"].isin(missing_times)].copy()



    full_edges = _build_full_edge_list(df)

    if not full_edges:

        print(f"[{file_stem}] empty dataset after preprocessing, skipping")

        return None



    tests_df_all = _parse_paths_tsv(file_stem)



    unique_times = sorted({int(t) for (_u, _v, t, _p) in full_edges})

    idx80 = min(max(int(0.8 * len(unique_times)), 0), len(unique_times) - 1)

    idx90 = min(max(int(0.9 * len(unique_times)), 0), len(unique_times) - 1)

    t80 = int(unique_times[idx80])

    t90 = int(unique_times[idx90])



    edges_80 = [(u, v, t, p) for (u, v, t, p) in full_edges if int(t) <= t80]

    edges_80_90 = [(u, v, t, p) for (u, v, t, p) in full_edges if t80 < int(t) <= t90]

    edges_90 = [(u, v, t, p) for (u, v, t, p) in full_edges if int(t) <= t90]



    vertex_list_80 = sorted({int(x) for (u, v, _t, _p) in edges_80 for x in (u, v)})

    vertex_list_90 = sorted({int(x) for (u, v, _t, _p) in edges_90 for x in (u, v)})

    vertex_set_90 = set(vertex_list_90)



    edges_after_90 = [

        (u, v, t, p)

        for (u, v, t, p) in full_edges

        if int(t) > t90 and int(u) in vertex_set_90 and int(v) in vertex_set_90

    ]

    future_times_test = sorted({int(t) for (_u, _v, t, _p) in edges_after_90})

    future_times_80_90 = sorted({int(t) for (_u, _v, t, _p) in edges_80_90})



    eligible_tests_df = tests_df_all[tests_df_all["time"] > t90].copy()

    if eligible_tests_df.empty:

        print(f"[{file_stem}] no eligible tests after t90, skipping")

        return None



    pivot_time = t90

    all_edges, pivot_edges, vertex_list, future_times, global_interval, pivot_interval = _build_streams(df, pivot_time)

    all_edges.sort(key=lambda e: int(e[2]))

    pivot_edges.sort(key=lambda e: int(e[2]))



    if not future_times:

        print(f"[{file_stem}] no future timestamps after pivot, skipping")

        return None



    pivot_observed_edges = {(int(u), int(v)) for (u, v, _t, _p) in pivot_edges}

    pivot_observed_out = defaultdict(set)

    for u, v in pivot_observed_edges:

        pivot_observed_out[int(u)].add(int(v))



    active_fallback_target_pool = _build_global_activity_pool_from_prefix(

        edge_list=pivot_edges,

        pool_size=cfg.dest_connector_active_fallback_size,

    )



    pair_times = build_pair_time_index(full_edges)



    return {

        "file_stem": file_stem,

        "dataset": DATASET_DISPLAY.get(file_stem, file_stem),

        "df": df,

        "full_edges": full_edges,

        "tests_df_all": tests_df_all,

        "eligible_tests_df": eligible_tests_df,

        "t80": t80,

        "t90": t90,

        "pivot_time": pivot_time,

        "edges_80": edges_80,

        "edges_80_90": edges_80_90,

        "edges_90": edges_90,

        "edges_after_90": edges_after_90,

        "future_times_test": future_times_test,

        "future_times_80_90": future_times_80_90,

        "vertex_list_80": vertex_list_80,

        "vertex_list_90": vertex_list_90,

        "all_edges": all_edges,

        "pivot_edges": pivot_edges,

        "vertex_list": vertex_list,

        "future_times": sorted(int(t) for t in future_times),

        "global_interval": global_interval,

        "pivot_interval": pivot_interval,

        "pivot_observed_edges": pivot_observed_edges,

        "pivot_observed_out": pivot_observed_out,

        "active_fallback_target_pool": active_fallback_target_pool,

        "pair_times": pair_times,

        "max_future_timing_horizon": max(int(t) for t in future_times),

        "landmarks_num": int(getattr(cfg, "num_landmarks", dataset_landmarks(file_stem))),

    }





def sample_iteration_tests(data: Dict[str, Any], cfg: PipelineConfig, iteration: int) -> pd.DataFrame:

    seed = iteration_query_seed(cfg, iteration)

    eligible = data["eligible_tests_df"]

    if len(eligible) > cfg.num_random_queries:

        return eligible.sample(n=cfg.num_random_queries, random_state=seed).reset_index(drop=True)

    return eligible.sample(frac=1.0, random_state=seed).reset_index(drop=True)





def train_static_oracle(

    data: Dict[str, Any],

    cfg: PipelineConfig,

    iteration: int,

    embeddings_dir: Path,

) -> Tuple[Dict[str, Any], Any, float, pd.DataFrame, pd.DataFrame]:

    seed = iteration_query_seed(cfg, iteration)

    file_stem = data["file_stem"]



    start = time_lib.perf_counter()

    embeddings_80 = _normalize_embeddings(_load_embeddings_for_prefix(

        file_stem,

        "80",

        data["edges_80"],

        embeddings_dir=embeddings_dir,

        seed=seed,

        force_recompute=cfg.force_recompute_embeddings_per_iteration,

    ))

    embeddings_90 = _normalize_embeddings(_load_embeddings_for_prefix(

        file_stem,

        "90",

        data["edges_90"],

        embeddings_dir=embeddings_dir,

        seed=seed,

        force_recompute=cfg.force_recompute_embeddings_per_iteration,

    ))

    embedding_s = time_lib.perf_counter() - start



    train_start = time_lib.perf_counter()

    oracle = _train_lp_oracle(

        mode="static",

        edges_train=data["edges_80"],

        edges_future=data["edges_80_90"],

        vertex_list=data["vertex_list_80"],

        embeddings_dict=embeddings_80,

        pivot_time=data["t80"],

        future_times_train=data["future_times_80_90"],

        seed=seed,

    )

    train_s = time_lib.perf_counter() - train_start + embedding_s



    best_threshold, val_df = _evaluate_lp_oracle_on_window(

        oracle=oracle,

        edges_prefix=data["edges_80"],

        edges_future=data["edges_80_90"],

        vertex_list=data["vertex_list_80"],

        embeddings_dict=embeddings_80,

        pivot_time=data["t80"],

        future_times_eval=data["future_times_80_90"],

        lp_edge_thresholds=list(cfg.static_edge_thresholds),

        seed=seed,

    )



    if data["edges_after_90"] and data["future_times_test"]:

        _unused, test_df = _evaluate_lp_oracle_on_window(

            oracle=oracle,

            edges_prefix=data["edges_90"],

            edges_future=data["edges_after_90"],

            vertex_list=data["vertex_list_90"],

            embeddings_dict=embeddings_90,

            pivot_time=data["t90"],

            future_times_eval=data["future_times_test"],

            lp_edge_thresholds=[best_threshold],

            seed=seed,

        )

    else:

        test_df = pd.DataFrame([{"threshold": best_threshold}])



    oracle["selected_threshold"] = float(best_threshold)

    oracle["embeddings_90"] = embeddings_90

    oracle["train_s"] = train_s

    return oracle, embeddings_90, float(best_threshold), val_df, test_df





def train_tgn_oracle(

    data: Dict[str, Any],

    cfg: PipelineConfig,

    iteration: int,

) -> Tuple[Dict[str, Any], float, pd.DataFrame, pd.DataFrame]:

    seed = iteration_query_seed(cfg, iteration)

    train_start = time_lib.perf_counter()

    oracle = _train_tgn_oracle(

        edges_train=data["edges_80"],

        edges_prefix_for_nodes=data["edges_90"],

        edges_val=data["edges_80_90"],

        seed=seed,

    )

    train_s = time_lib.perf_counter() - train_start



    _warmup_tgn_oracle(oracle, data["edges_80"])

    best_threshold, val_df = _evaluate_tgn_oracle_on_window_official_update(

        oracle=oracle,

        edges_future=data["edges_80_90"],

        negative_source_edges=data["edges_90"],

        neighbor_source_edges=data["edges_90"],

        lp_edge_thresholds=list(cfg.tgn_edge_thresholds),

        seed=seed,

    )



    _warmup_tgn_oracle(oracle, data["edges_90"])

    if data["edges_after_90"] and data["future_times_test"]:

        _unused, test_df = _evaluate_tgn_oracle_on_window_official_update(

            oracle=oracle,

            edges_future=data["edges_after_90"],

            negative_source_edges=data["full_edges"],

            neighbor_source_edges=data["full_edges"],

            lp_edge_thresholds=[best_threshold],

            seed=seed,

        )

        _warmup_tgn_oracle(oracle, data["edges_90"])

    else:

        test_df = pd.DataFrame([{"threshold": best_threshold}])



    oracle["selected_threshold"] = float(best_threshold)

    oracle["train_s"] = train_s

    return oracle, float(best_threshold), val_df, test_df





def train_jodie_oracle(

    data: Dict[str, Any],

    cfg: PipelineConfig,

    iteration: int,

) -> Tuple[Dict[str, Any], float, pd.DataFrame, pd.DataFrame]:

    seed = iteration_query_seed(cfg, iteration)

    oracle = _train_jodie_oracle(

        logic=cfg.jodie_logic,

        edges_80=data["edges_80"],

        edges_80_90=data["edges_80_90"],

        edges_90=data["edges_90"],

        vertex_list_80=data["vertex_list_80"],

        future_times_80_90=data["future_times_80_90"],

        seed=seed,

        jodie_epochs=cfg.jodie_epochs,

        jodie_embedding_dim=cfg.jodie_embedding_dim,

        jodie_lr=cfg.jodie_lr,

        jodie_weight_decay=cfg.jodie_weight_decay,

        jodie_topk_per_source=cfg.jodie_path_topk_per_source,

    )



    _jodie_warmup_state(oracle, data["edges_80"])

    best_threshold, val_df = _evaluate_jodie_oracle_on_window(

        oracle=oracle,

        edges_future=data["edges_80_90"],

        vertex_list=data["vertex_list_80"],

        future_times_eval=data["future_times_80_90"],

        lp_edge_thresholds=list(cfg.jodie_edge_thresholds),

        topk_per_source=cfg.jodie_eval_topk_per_source,

        seed=seed,

        label_suffix="validation/calibration",

    )



    _jodie_warmup_state(oracle, data["edges_90"])

    if data["edges_after_90"] and data["future_times_test"]:

        _unused, test_df = _evaluate_jodie_oracle_on_window(

            oracle=oracle,

            edges_future=data["edges_after_90"],

            vertex_list=data["vertex_list_90"],

            future_times_eval=data["future_times_test"],

            lp_edge_thresholds=[best_threshold],

            topk_per_source=cfg.jodie_eval_topk_per_source,

            seed=seed,

            label_suffix="test",

        )

        _jodie_warmup_state(oracle, data["edges_90"])

    else:

        test_df = pd.DataFrame([{"threshold": best_threshold}])



    oracle["selected_threshold"] = float(best_threshold)

    return oracle, float(best_threshold), val_df, test_df





def run_static_or_tgn_pipeline(

    data: Dict[str, Any],

    cfg: PipelineConfig,

    method: str,

    oracle_mode: str,

    oracle: Dict[str, Any],

    embeddings_90: Any,

    lp_edge_threshold: float,

    tests_df_iter: pd.DataFrame,

    iteration: int,

) -> List[Dict[str, Any]]:

    pivot_time = int(data["pivot_time"])

    tests_by_src = _group_tests_by_src(tests_df_iter)

    active_fallback_target_pool = data["active_fallback_target_pool"]

    pivot_edges = data["pivot_edges"]

    future_times = data["future_times"]



    workload_landmark_pool = _build_prefix_supported_landmark_pool(

        tests_df=tests_df_iter,

        edge_list=pivot_edges,

        pool_size=data["landmarks_num"],

        connector_pool_size=cfg.dest_connector_pool_size,

        active_fallback_pool=active_fallback_target_pool,

        active_fallback_size=cfg.dest_connector_active_fallback_size,

    )

    landmark_connector_target_pool = _union_destination_connector_pools_from_prefix(

        edge_list=pivot_edges,

        destinations=workload_landmark_pool,

        connector_pool_size=cfg.landmark_connector_pool_size,

        active_fallback_pool=active_fallback_target_pool,

        active_fallback_size=0,

    )



    detail_rows: List[Dict[str, Any]] = []

    prediction_rows: List[Dict[str, Any]] = []

    timing_rows: List[Dict[str, Any]] = []

    accepted_edge_rows: List[Dict[str, Any]] = []

    destination_connector_pool_cache: Dict[int, set] = {}

    edge_score_cache: Dict[Any, float] = {}

    edge_best_time_cache: Dict[Any, Any] = {}

    edge_cache_low_by_tu = defaultdict(set)

    edge_cache_high_by_tu = defaultdict(dict)



    for src, src_rows in tests_by_src.items():

        src = int(src)

        all_edges = list(data["all_edges"])

        pivot_edges_src = list(data["pivot_edges"])

        all_edges.sort(key=lambda e: (int(e[2]), int(e[0]) != src))

        pivot_edges_src.sort(key=lambda e: (int(e[2]), int(e[0]) != src))



        (

            sp_pivot_base,

            L_pivot_base,

            _sp_pivot2_base,

            actual_pivot_base,

            dest_paths_pivot_base,

        ) = RES.computeActualShortestPathAndDistance(

            pivot_edges_src,

            src,

            data["vertex_list"],

            data["pivot_interval"],

        )

        pivot_baseline_sp_base = dict(sp_pivot_base)



        for row in src_rows:

            dst = int(row.destination)

            original_fut_time = int(row.time)

            if oracle_mode == "tgn" and cfg.tgn_edge_time_selection_mode == "best_score_until_query_time":

                fut_time = original_fut_time

            elif cfg.run_timing_to_max_future_timestamp:

                fut_time = int(data["max_future_timing_horizon"])

            else:

                fut_time = original_fut_time



            (

                sp_pivot,

                L_pivot,

                actual_pivot,

                dest_paths_pivot,

                pivot_baseline_sp,

            ) = _fresh_query_state(

                sp_pivot_base,

                L_pivot_base,

                actual_pivot_base,

                dest_paths_pivot_base,

                pivot_baseline_sp_base,

            )



            pivot_entry = pivot_baseline_sp.get(dst, (math.inf, [], 0, 0.0))

            pivot_nodes = pivot_entry[1]

            reference_len = len(pivot_nodes) - 1 if pivot_nodes else None

            pivot_dist = pivot_entry[0]

            pivot_tform = pivot_entry[2] if len(pivot_entry) > 2 else pivot_time

            if pivot_dist != math.inf and pivot_nodes:

                last_best_tuple = (pivot_dist, pivot_tform, pivot_nodes, [], 1.0, 1.0)

            else:

                last_best_tuple = (math.inf, math.inf, [], [], 0.0, 0.0)

            answer_tuple = last_best_tuple

            answer_topk_tuples = [answer_tuple] if answer_tuple[2] else []



            if dst not in destination_connector_pool_cache:

                destination_connector_pool_cache[dst] = _build_destination_connector_target_pool_from_prefix(

                    edge_list=pivot_edges_src,

                    dst=dst,

                    connector_pool_size=cfg.dest_connector_pool_size,

                    active_fallback_pool=active_fallback_target_pool,

                    active_fallback_size=cfg.dest_connector_active_fallback_size,

                )

            destination_connector_target_pool = destination_connector_pool_cache[dst]



            prev_time = pivot_time

            accepted_predicted_out = defaultdict(set)

            compiled_candidate_index_cache: Dict[Any, Any] = {}

            path_state_version = 0

            scheduled_predicted_edges_by_time = defaultdict(list)

            scheduled_predicted_edge_keys = set()

            accepted_records: List[Dict[str, Any]] = []



            for t_cur in future_times:

                t_cur = int(t_cur)

                if t_cur <= prev_time:

                    continue

                if t_cur > fut_time:

                    break



                _pipeline_step_start = time_lib.perf_counter()

                build_params = inspect.signature(_build_candidate_pairs_fast_with_cache).parameters

                use_score_cache_for_materialization = (

                    cfg.use_cache

                    and not (

                        oracle_mode == "tgn"

                        and cfg.tgn_edge_time_selection_mode == "best_score_until_query_time"

                        and cfg.disable_materialization_score_cache_in_best_time_mode

                    )

                )



                if "edge_cache_low_by_tu" in build_params:

                    (

                        candidate_pair_to_best_len,

                        cached_scored_candidate_occurrences,

                        missing_candidate_occurrences_by_pair,

                        _candidate_build_stats,

                    ) = _build_candidate_pairs_fast_with_cache(

                        vertexToLv_dict=L_pivot,

                        dst=dst,

                        reference_len=reference_len,

                        observed_out=data["pivot_observed_out"],

                        accepted_out=accepted_predicted_out,

                        t_cur=t_cur,

                        edge_score_cache=edge_score_cache,

                        edge_cache_low_by_tu=edge_cache_low_by_tu,

                        edge_cache_high_by_tu=edge_cache_high_by_tu,

                        lp_edge_threshold=lp_edge_threshold,

                        oracle_mode=oracle_mode,

                        compiled_candidate_index_cache=compiled_candidate_index_cache,

                        path_state_version=path_state_version,

                        use_edge_score_cache=use_score_cache_for_materialization,

                        use_candidate_path_index_cache=cfg.use_cache,

                        active_target_pool=destination_connector_target_pool,

                        landmark_target_pool=workload_landmark_pool,

                        landmark_connector_target_pool=landmark_connector_target_pool,

                        landmark_nodes=workload_landmark_pool,

                        root_source=src,

                    )

                else:

                    (

                        candidate_pair_to_best_len,

                        cached_scored_candidate_occurrences,

                        missing_candidate_occurrences_by_pair,

                        _candidate_build_stats,

                    ) = _build_candidate_pairs_fast_with_cache(

                        vertexToLv_dict=L_pivot,

                        dst=dst,

                        reference_len=reference_len,

                        observed_out=data["pivot_observed_out"],

                        accepted_out=accepted_predicted_out,

                        edge_score_cache=edge_score_cache,

                        edge_cache_low_by_u=edge_cache_low_by_tu,

                        edge_cache_high_by_u=edge_cache_high_by_tu,

                        lp_edge_threshold=lp_edge_threshold,

                        compiled_candidate_index_cache=compiled_candidate_index_cache,

                        path_state_version=path_state_version,

                        use_edge_score_cache=use_score_cache_for_materialization,

                        use_candidate_path_index_cache=cfg.use_cache,

                        active_target_pool=destination_connector_target_pool,

                        landmark_target_pool=workload_landmark_pool,

                        landmark_connector_target_pool=landmark_connector_target_pool,

                        landmark_nodes=workload_landmark_pool,

                        root_source=src,

                    )



                selected_time_by_pair: Dict[Tuple[int, int], int] = {}

                _edge_stats: Dict[str, Any] = {}

                if oracle_mode == "tgn" and cfg.tgn_edge_time_selection_mode == "best_score_until_query_time":

                    candidate_times = _candidate_time_grid_until_query_time(

                        future_times=future_times,

                        t_start=t_cur,

                        query_future_time=original_fut_time,

                        max_edge_time_candidates=cfg.max_edge_time_candidates,

                    )

                    new_scored_candidates, selected_time_by_pair, _edge_stats = _score_candidate_pairs_best_time_until_query_cached(

                        oracle=oracle,

                        candidate_pair_to_best_len=candidate_pair_to_best_len,

                        candidate_times=candidate_times,

                        pivot_time_inference=pivot_time,

                        embeddings_dict=embeddings_90,

                        edge_score_cache=edge_score_cache,

                        edge_cache_low_by_tu=edge_cache_low_by_tu,

                        edge_cache_high_by_tu=edge_cache_high_by_tu,

                        lp_edge_threshold=lp_edge_threshold,

                        oracle_mode=oracle_mode,

                        edge_best_time_cache=edge_best_time_cache,

                        use_edge_score_cache=cfg.use_cache,

                        use_edge_best_time_cache=cfg.use_cache,

                    )

                else:

                    new_scored_candidates = _score_candidate_pairs_batch(

                        oracle=oracle,

                        candidate_pair_to_best_len=candidate_pair_to_best_len,

                        t_cur=t_cur,

                        pivot_time_inference=pivot_time,

                        embeddings_dict=embeddings_90,

                    )

                    if cfg.use_cache:

                        for u_new, v_new, _new_len, p_hat_new in new_scored_candidates:

                            store_params = inspect.signature(_store_edge_score_in_indexes).parameters

                            if "edge_cache_low_by_tu" in store_params:

                                _store_edge_score_in_indexes(

                                    edge_score_cache=edge_score_cache,

                                    edge_cache_low_by_tu=edge_cache_low_by_tu,

                                    edge_cache_high_by_tu=edge_cache_high_by_tu,

                                    oracle_mode=oracle_mode,

                                    u=u_new,

                                    v=v_new,

                                    t_cur=t_cur,

                                    p_hat=p_hat_new,

                                    lp_edge_threshold=lp_edge_threshold,

                                )

                            else:

                                _store_edge_score_in_indexes(

                                    edge_score_cache=edge_score_cache,

                                    edge_cache_low_by_u=edge_cache_low_by_tu,

                                    edge_cache_high_by_u=edge_cache_high_by_tu,

                                    u=u_new,

                                    v=v_new,

                                    p_hat=p_hat_new,

                                    lp_edge_threshold=lp_edge_threshold,

                                )

                            selected_time_by_pair[(int(u_new), int(v_new))] = int(t_cur)



                new_score_by_pair = {

                    (int(u), int(v)): float(p_hat)

                    for u, v, _new_len, p_hat in new_scored_candidates

                }

                new_scored_occurrences = []

                for pair_key, occurrences in missing_candidate_occurrences_by_pair.items():

                    p_hat = new_score_by_pair.get(pair_key)

                    if p_hat is None:

                        continue

                    for lv_path_id, u, v, new_len in occurrences:

                        new_scored_occurrences.append((int(lv_path_id), int(u), int(v), int(new_len), float(p_hat)))



                for _lv, u_cached, v_cached, _new_len, _p in cached_scored_candidate_occurrences:

                    selected_time_by_pair[(int(u_cached), int(v_cached))] = int(t_cur)



                scored_candidates = list(cached_scored_candidate_occurrences) + new_scored_occurrences

                accepted_before_policy = [cand for cand in scored_candidates if float(cand[4]) >= float(lp_edge_threshold)]

                if cfg.disable_root_direct_edges:

                    is_root_direct = getattr(

                        ns,

                        "_is_root_direct_candidate",

                        lambda cand, dst_value: int(cand[2]) == int(dst_value) and int(cand[3]) == 1,

                    )

                    accepted_candidates = [cand for cand in accepted_before_policy if not is_root_direct(cand, dst)]

                else:

                    accepted_candidates = list(accepted_before_policy)



                cap_k = cfg.max_accepted_edges_per_lv_path

                if cap_k is not None and int(cap_k) > 0:

                    accepted_by_lv_path = defaultdict(list)

                    for cand in accepted_candidates:

                        accepted_by_lv_path[int(cand[0])].append(cand)

                    limited = []

                    for _lv_path_id, group in accepted_by_lv_path.items():

                        topk_key = getattr(

                            ns,

                            "_accepted_candidate_topk_key",

                            lambda x: (-float(x[4]), int(x[3]), int(x[1]), int(x[2])),

                        )

                        group = sorted(group, key=topk_key)

                        limited.extend(group[: int(cap_k)])

                    accepted_candidates = limited



                pred_stream = []

                predicted_edges_this_time = set()



                due_release_times = [

                    int(t_release)

                    for t_release in list(scheduled_predicted_edges_by_time.keys())

                    if int(t_release) <= int(t_cur)

                ]

                for t_release in sorted(due_release_times):

                    for u_s, v_s, t_s, p_s in scheduled_predicted_edges_by_time.pop(t_release, []):

                        edge_key = (int(u_s), int(v_s), int(t_s))

                        if edge_key in predicted_edges_this_time:

                            continue

                        pred_stream.append((int(u_s), int(v_s), int(t_s), float(p_s)))

                        predicted_edges_this_time.add(edge_key)



                for lv_path_id, u, v, new_len, p_hat in accepted_candidates:

                    accepted_predicted_out[int(u)].add(int(v))

                    selected_edge_time = int(selected_time_by_pair.get((int(u), int(v)), int(t_cur)))

                    edge_key = (int(u), int(v), selected_edge_time)

                    if edge_key not in scheduled_predicted_edge_keys:

                        scheduled_predicted_edge_keys.add(edge_key)

                        accepted_records.append({

                            "u": int(u),

                            "v": int(v),

                            "time": selected_edge_time,

                            "score": float(p_hat),

                            "new_len": int(new_len),

                            "lv_path_id": int(lv_path_id),

                        })

                    if selected_edge_time <= int(t_cur):

                        if edge_key not in predicted_edges_this_time:

                            pred_stream.append((int(u), int(v), selected_edge_time, float(p_hat)))

                            predicted_edges_this_time.add(edge_key)

                    else:

                        scheduled_predicted_edges_by_time[selected_edge_time].append(

                            (int(u), int(v), selected_edge_time, float(p_hat))

                        )



                prev_time = t_cur



                (

                    shortest_dict,

                    L_dict,

                    actual_dict,

                    dest_paths_dict,

                    best_sp_dict,

                    topk_sp_dict,

                    _candidate_count_dict,

                    _timing_info,

                ) = PredictSP(

                    pred_stream,

                    L_pivot,

                    sp_pivot,

                    actual_pivot,

                    src,

                    data["vertex_list"],

                    (data["pivot_interval"][0], t_cur),

                    dest_paths_pivot,

                    cfg.path_exist_thresholds[0],

                    cfg.shortest_proba_bounds[0],

                    cfg.lk_runs,

                    pivot_time=pivot_time,

                    pivot_baseline_sp=pivot_baseline_sp,

                    return_timing=True,

                    top_k_result_paths=cfg.top_k_result_paths,

                    return_topk=True,

                )



                sp_pivot, L_pivot, actual_pivot, dest_paths_pivot = shortest_dict, L_dict, actual_dict, dest_paths_dict

                if accepted_candidates:

                    path_state_version += 1



                best_tuple = best_sp_dict.get(dst, (math.inf, math.inf, [], [], 0.0, 0.0))

                topk_answer_tuples = list(topk_sp_dict.get(dst, []))

                if not topk_answer_tuples and best_tuple[2]:

                    topk_answer_tuples = [best_tuple]



                d_val, t_form, path_nodes, path_edges, p_sp, p_exist = best_tuple

                if reference_len is None and d_val != math.inf and path_nodes:

                    reference_len = len(path_nodes) - 1



                answer_tuple = best_tuple

                answer_topk_tuples = topk_answer_tuples

                timing_rows.append(_timing_row_for_step(

                    dataset=data["dataset"],

                    file_stem=data["file_stem"],

                    iteration=iteration,

                    method=method,

                    row=row,

                    pivot_time=pivot_time,

                    t_cur=t_cur,

                    step_total_s=time_lib.perf_counter() - _pipeline_step_start,

                    candidate_pair_to_best_len=candidate_pair_to_best_len,

                    cached_scored_candidate_occurrences=cached_scored_candidate_occurrences,

                    missing_candidate_occurrences_by_pair=missing_candidate_occurrences_by_pair,

                    new_scored_candidates=new_scored_candidates,

                    scored_candidates=scored_candidates,

                    accepted_before_policy=accepted_before_policy,

                    accepted_candidates=accepted_candidates,

                    candidate_build_stats=_candidate_build_stats,

                    edge_stats=_edge_stats,

                    candidate_count_dict=_candidate_count_dict,

                    timing_info=_timing_info,

                ))



                if cfg.freeze_answer_at_original_future_time and t_cur >= original_fut_time:

                    break



            prediction_rows.extend(_topk_prediction_rows_for_query(

                dataset=data["dataset"],

                file_stem=data["file_stem"],

                iteration=iteration,

                method=method,

                row=row,

                pivot_time=pivot_time,

                topk_tuples=answer_topk_tuples,

                max_rank=cfg.top_k_result_paths,

            ))

            accepted_edge_rows.extend(_accepted_edge_rows_for_query(

                dataset=data["dataset"],

                file_stem=data["file_stem"],

                iteration=iteration,

                method=method,

                row=row,

                pivot_time=pivot_time,

                accepted_edges=accepted_records,

            ))



            detail_rows.append(compute_query_result(

                dataset=data["dataset"],

                iteration=iteration,

                method=method,

                row=row,

                pivot_time=pivot_time,

                pair_times=data["pair_times"],

                accepted_edges=accepted_records,

                selected_tuple=answer_tuple,

            ))



    _write_method_artifacts(

        cfg=cfg,

        data=data,

        iteration=iteration,

        method=method,

        prediction_rows=prediction_rows,

        timing_rows=timing_rows,

        accepted_edge_rows=accepted_edge_rows,

    )



    return detail_rows





DATASETS = [

    ("ia-enron-employees_TimestampZero_NoDuplicate_sorted.csv", -1),

    ("email-Eu-core-temporal_TimestampZero_NoDuplicate_sorted.csv", -1),

    ("CollegeMsg_TimestampZero_NoDuplicate_sorted.csv", -1),

    ("ml_bitcoinotc_disperse_NoDuplicate_sorted.csv", -1),

]



DATASET_DISPLAY = {

    "CollegeMsg_TimestampZero_NoDuplicate_sorted": "CollegeMsg",

    "ia-enron-employees_TimestampZero_NoDuplicate_sorted": "Enron",

    "email-Eu-core-temporal_TimestampZero_NoDuplicate_sorted": "Email-Eu",

    "ml_bitcoinotc_disperse_NoDuplicate_sorted": "Bitcoin",

}

METHOD_ORDER = ["N2VLP-Static", "TGN-MaxTime", "TGN-PerTime", "Jodie-Frozen", "Jodie-Update"]

DATASET_ORDER = ["CollegeMsg", "Enron", "Email-Eu", "Bitcoin"]





def PipelineConfig(**kwargs):

    defaults = {

        "tests_dir": Path("Data/query_tests"),

        "test_suffix": "",

        "output_dir": Path("Results"),

        "num_random_queries": 100,

        "num_dataset_iterations": 1,

        "query_sample_seed": 42,

        "path_exist_thresholds": (0.0,),

        "shortest_proba_bounds": (0.0,),

        "lk_runs": 20,

        "top_k_result_paths": 10,

        "static_edge_thresholds": (0.5,),

        "tgn_edge_thresholds": (0.5,),

        "jodie_edge_thresholds": (0.1,),

        "use_cache": True,

        "force_recompute_embeddings_per_iteration": True,

        "train_lp_once_per_dataset": True,

        "train_tgn_once_per_dataset": True,

        "train_jodie_once_per_dataset": True,

        "run_timing_to_max_future_timestamp": True,

        "freeze_answer_at_original_future_time": True,

        "use_fast_state_clone": False,

        "block_accepted_predicted_edges": False,

        "tgn_edge_time_selection_mode": "best_score_until_query_time",

        "max_edge_time_candidates": None,

        "disable_materialization_score_cache_in_best_time_mode": True,

        "jodie_logic": "jodie_rr",

        "jodie_eval_topk_per_source": 100,

        "jodie_path_topk_per_source": 10,

        "jodie_epochs": 50,

        "jodie_embedding_dim": 128,

        "jodie_lr": 1e-3,

        "jodie_weight_decay": 1e-5,

        "jodie_self_update_top_rank": 1,

        "jodie_self_update_order": "score_desc",

        "dest_connector_pool_size": 50,

        "dest_connector_active_fallback_size": 20,

        "dest_connector_source_budget": 5,

        "landmark_connector_pool_size": 50,

        "landmark_connector_weight": 2.0,

        "landmark_prefix_path_weight": 1.0,

        "landmark_reachability_weight": 0.25,

        "landmark_destination_frequency_weight": 0.0,

        "landmark_reachability_max_distance": 3,

        "max_accepted_edges_per_lv_path": None,

        "disable_root_direct_edges": False,

    }

    defaults.update(kwargs)

    return SimpleNamespace(**defaults)



def apply_config_globals(cfg: PipelineConfig, oracle_mode: str = "static", jodie_state_mode: str = "frozen") -> None:

    global TESTS_DIR, TOP100_TEST_SUFFIX, NUM_RANDOM_QUERIES, NUM_DATASET_ITERATIONS, QUERY_SAMPLE_SEED

    global PATH_EXIST_THRESHOLDS, SHORTEST_PROBA_BOUNDS, LK_RUNS, TOP_K_RESULT_PATHS, LP_EDGE_THRESHOLDS, LP_ORACLE_MODE

    global RUN_TIMING_TO_MAX_FUTURE_TIMESTAMP, FREEZE_ANSWER_AT_ORIGINAL_FUT_TIME, USE_FAST_STATE_CLONE

    global FORCE_RECOMPUTE_EMBEDDINGS_PER_ITERATION, BLOCK_ACCEPTED_PREDICTED_EDGES

    global USE_CACHE, USE_EDGE_SCORE_CACHE, EDGE_TIME_SELECTION_MODE, MAX_EDGE_TIME_CANDIDATES

    global DISABLE_MATERIALIZATION_SCORE_CACHE_IN_BEST_TIME_MODE, USE_EDGE_BEST_TIME_CACHE

    global USE_CANDIDATE_PATH_INDEX_CACHE, USE_PATH_CANDIDATE_PAIR_CACHE, USE_JODIE_SCORE_CACHE

    global USE_DESTINATION_CONNECTOR_TARGETS, DEST_CONNECTOR_POOL_SIZE, DEST_CONNECTOR_ACTIVE_FALLBACK_SIZE

    global DEST_CONNECTOR_SOURCE_BUDGET, USE_PREFIX_SUPPORTED_LANDMARKS, USE_WORKLOAD_LANDMARKS

    global LANDMARK_CONNECTOR_POOL_SIZE, LANDMARK_CONNECTOR_WEIGHT, LANDMARK_PREFIX_PATH_WEIGHT

    global LANDMARK_REACHABILITY_WEIGHT, LANDMARK_DESTINATION_FREQUENCY_WEIGHT, LANDMARK_REACHABILITY_MAX_DISTANCE

    global INCLUDE_CONNECTOR_SOURCES_AS_CANDIDATE_SOURCES, INCLUDE_LANDMARK_CONNECTOR_SOURCES

    global INCLUDE_REACHED_LANDMARK_SOURCES, INCLUDE_PATH_TARGETS_IN_CONNECTOR_MODE, INCLUDE_REACHED_NODES_AS_COMPLETION_SOURCES

    global USE_ACTIVE_TARGET_POOL, ACTIVE_TARGET_POOL_SIZE, USE_ACTIVE_TARGETS_ONLY_FOR_TOP_SOURCES, ACTIVE_TARGET_SOURCE_BUDGET

    global DISABLE_ROOT_DIRECT_EDGES, MAX_ACCEPTED_EDGES_PER_LV_PATH

    global JODIE_EVAL_TOPK_PER_SOURCE, JODIE_PATH_TOPK_PER_SOURCE, JODIE_EPOCHS, JODIE_EMBEDDING_DIM

    global JODIE_LR, JODIE_WEIGHT_DECAY, JODIE_PATH_STATE_MODE, JODIE_SELF_UPDATE_TOP_RANK, JODIE_SELF_UPDATE_ORDER

    global JODIE_RR_SCORE_MODE, JODIE_ONLINE_EVAL_UPDATE_TRUE_EDGES



    TESTS_DIR = Path(cfg.tests_dir)

    TOP100_TEST_SUFFIX = cfg.test_suffix

    NUM_RANDOM_QUERIES = int(cfg.num_random_queries)

    NUM_DATASET_ITERATIONS = int(cfg.num_dataset_iterations)

    QUERY_SAMPLE_SEED = cfg.query_sample_seed

    PATH_EXIST_THRESHOLDS = list(cfg.path_exist_thresholds)

    SHORTEST_PROBA_BOUNDS = list(cfg.shortest_proba_bounds)

    LK_RUNS = int(cfg.lk_runs)

    TOP_K_RESULT_PATHS = int(cfg.top_k_result_paths)

    LP_ORACLE_MODE = oracle_mode



    if oracle_mode == "tgn":

        LP_EDGE_THRESHOLDS = list(cfg.tgn_edge_thresholds)

        EDGE_TIME_SELECTION_MODE = cfg.tgn_edge_time_selection_mode

        USE_EDGE_SCORE_CACHE = bool(cfg.use_cache)

        USE_EDGE_BEST_TIME_CACHE = bool(cfg.use_cache)

    elif oracle_mode == "static":

        LP_EDGE_THRESHOLDS = list(cfg.static_edge_thresholds)

        EDGE_TIME_SELECTION_MODE = "current_time"

        USE_EDGE_SCORE_CACHE = bool(cfg.use_cache)

        USE_EDGE_BEST_TIME_CACHE = False

    else:

        LP_EDGE_THRESHOLDS = list(cfg.jodie_edge_thresholds)

        EDGE_TIME_SELECTION_MODE = "current_time"

        USE_EDGE_SCORE_CACHE = False

        USE_EDGE_BEST_TIME_CACHE = False



    RUN_TIMING_TO_MAX_FUTURE_TIMESTAMP = bool(cfg.run_timing_to_max_future_timestamp)

    FREEZE_ANSWER_AT_ORIGINAL_FUT_TIME = bool(cfg.freeze_answer_at_original_future_time)

    USE_FAST_STATE_CLONE = bool(cfg.use_fast_state_clone)

    FORCE_RECOMPUTE_EMBEDDINGS_PER_ITERATION = bool(cfg.force_recompute_embeddings_per_iteration)

    BLOCK_ACCEPTED_PREDICTED_EDGES = bool(cfg.block_accepted_predicted_edges)

    USE_CACHE = bool(cfg.use_cache)

    MAX_EDGE_TIME_CANDIDATES = cfg.max_edge_time_candidates

    DISABLE_MATERIALIZATION_SCORE_CACHE_IN_BEST_TIME_MODE = bool(cfg.disable_materialization_score_cache_in_best_time_mode)

    USE_CANDIDATE_PATH_INDEX_CACHE = bool(cfg.use_cache)

    USE_PATH_CANDIDATE_PAIR_CACHE = bool(cfg.use_cache)

    USE_JODIE_SCORE_CACHE = False



    USE_DESTINATION_CONNECTOR_TARGETS = True

    DEST_CONNECTOR_POOL_SIZE = int(cfg.dest_connector_pool_size)

    DEST_CONNECTOR_ACTIVE_FALLBACK_SIZE = int(cfg.dest_connector_active_fallback_size)

    DEST_CONNECTOR_SOURCE_BUDGET = int(cfg.dest_connector_source_budget)

    USE_PREFIX_SUPPORTED_LANDMARKS = True

    USE_WORKLOAD_LANDMARKS = True

    LANDMARK_CONNECTOR_POOL_SIZE = int(cfg.landmark_connector_pool_size)

    LANDMARK_CONNECTOR_WEIGHT = float(cfg.landmark_connector_weight)

    LANDMARK_PREFIX_PATH_WEIGHT = float(cfg.landmark_prefix_path_weight)

    LANDMARK_REACHABILITY_WEIGHT = float(cfg.landmark_reachability_weight)

    LANDMARK_DESTINATION_FREQUENCY_WEIGHT = float(cfg.landmark_destination_frequency_weight)

    LANDMARK_REACHABILITY_MAX_DISTANCE = int(cfg.landmark_reachability_max_distance)

    INCLUDE_CONNECTOR_SOURCES_AS_CANDIDATE_SOURCES = True

    INCLUDE_LANDMARK_CONNECTOR_SOURCES = True

    INCLUDE_REACHED_LANDMARK_SOURCES = True

    INCLUDE_PATH_TARGETS_IN_CONNECTOR_MODE = False

    INCLUDE_REACHED_NODES_AS_COMPLETION_SOURCES = True

    USE_ACTIVE_TARGET_POOL = False

    ACTIVE_TARGET_POOL_SIZE = 0

    USE_ACTIVE_TARGETS_ONLY_FOR_TOP_SOURCES = True

    ACTIVE_TARGET_SOURCE_BUDGET = int(cfg.dest_connector_source_budget)

    DISABLE_ROOT_DIRECT_EDGES = bool(cfg.disable_root_direct_edges)

    MAX_ACCEPTED_EDGES_PER_LV_PATH = cfg.max_accepted_edges_per_lv_path



    JODIE_EVAL_TOPK_PER_SOURCE = int(cfg.jodie_eval_topk_per_source)

    JODIE_PATH_TOPK_PER_SOURCE = int(cfg.jodie_path_topk_per_source)

    JODIE_EPOCHS = int(cfg.jodie_epochs)

    JODIE_EMBEDDING_DIM = int(cfg.jodie_embedding_dim)

    JODIE_LR = float(cfg.jodie_lr)

    JODIE_WEIGHT_DECAY = float(cfg.jodie_weight_decay)

    JODIE_PATH_STATE_MODE = jodie_state_mode

    JODIE_SELF_UPDATE_TOP_RANK = int(cfg.jodie_self_update_top_rank)

    JODIE_SELF_UPDATE_ORDER = cfg.jodie_self_update_order

    JODIE_RR_SCORE_MODE = "reciprocal_rank"

    JODIE_ONLINE_EVAL_UPDATE_TRUE_EDGES = True





def dataset_landmarks(file_stem: str) -> int:

    if file_stem in {"ia-enron-employees_TimestampZero_NoDuplicate_sorted", "email-Eu-core-temporal_TimestampZero_NoDuplicate_sorted"}:

        return 5

    if file_stem in {"CollegeMsg_TimestampZero_NoDuplicate_sorted", "ml_bitcoinotc_disperse_NoDuplicate_sorted"}:

        return 1

    return 1





def iteration_query_seed(cfg: PipelineConfig, iteration: int) -> Optional[int]:

    if cfg.query_sample_seed is None:

        return None

    return int(cfg.query_sample_seed) + int(iteration) - 1





def run_jodie_pipeline(

    data: Dict[str, Any],

    cfg: PipelineConfig,

    method: str,

    state_mode: str,

    oracle_frozen: Dict[str, Any],

    lp_edge_threshold: float,

    tests_df_iter: pd.DataFrame,

    iteration: int,

) -> List[Dict[str, Any]]:

    apply_config_globals(cfg, oracle_mode="jodie_rr", jodie_state_mode=state_mode)

    pivot_time = int(data["pivot_time"])

    tests_by_src = _group_tests_by_src(tests_df_iter)

    active_fallback_target_pool = data["active_fallback_target_pool"]

    pivot_edges = data["pivot_edges"]



    workload_landmark_pool = _build_prefix_supported_landmark_pool(

        tests_df=tests_df_iter,

        edge_list=pivot_edges,

        pool_size=data["landmarks_num"],

        connector_pool_size=cfg.dest_connector_pool_size,

        active_fallback_pool=active_fallback_target_pool,

        active_fallback_size=cfg.dest_connector_active_fallback_size,

    )

    landmark_connector_target_pool = _union_destination_connector_pools_from_prefix(

        edge_list=pivot_edges,

        destinations=workload_landmark_pool,

        connector_pool_size=cfg.landmark_connector_pool_size,

        active_fallback_pool=active_fallback_target_pool,

        active_fallback_size=0,

    )



    detail_rows: List[Dict[str, Any]] = []

    prediction_rows: List[Dict[str, Any]] = []

    timing_rows: List[Dict[str, Any]] = []

    accepted_edge_rows: List[Dict[str, Any]] = []

    destination_connector_pool_cache: Dict[int, set] = {}



    for src, src_rows in tests_by_src.items():

        src = int(src)

        pivot_edges_src = list(data["pivot_edges"])

        pivot_edges_src.sort(key=lambda e: (int(e[2]), int(e[0]) != src))



        (

            sp_pivot_base,

            L_pivot_base,

            _sp_pivot2_base,

            actual_pivot_base,

            dest_paths_pivot_base,

        ) = RES.computeActualShortestPathAndDistance(

            pivot_edges_src,

            src,

            data["vertex_list"],

            data["pivot_interval"],

        )

        pivot_baseline_sp_base = dict(sp_pivot_base)



        for row in src_rows:

            dst = int(row.destination)

            original_fut_time = int(row.time)

            fut_time = data["max_future_timing_horizon"] if cfg.run_timing_to_max_future_timestamp else original_fut_time



            (

                sp_pivot,

                L_pivot,

                actual_pivot,

                dest_paths_pivot,

                pivot_baseline_sp,

            ) = _fresh_query_state(

                sp_pivot_base,

                L_pivot_base,

                actual_pivot_base,

                dest_paths_pivot_base,

                pivot_baseline_sp_base,

            )



            oracle_query = _clone_jodie_runtime_oracle(oracle_frozen) if state_mode == "self_update" else oracle_frozen

            pivot_entry = pivot_baseline_sp.get(dst, (math.inf, [], 0, 0.0))

            pivot_nodes = pivot_entry[1]

            reference_len = len(pivot_nodes) - 1 if pivot_nodes else None

            pivot_dist = pivot_entry[0]

            pivot_tform = pivot_entry[2] if len(pivot_entry) > 2 else pivot_time

            answer_tuple = (

                (pivot_dist, pivot_tform, pivot_nodes, [], 1.0, 1.0)

                if pivot_dist != math.inf and pivot_nodes

                else (math.inf, math.inf, [], [], 0.0, 0.0)

            )

            answer_topk_tuples = [answer_tuple] if answer_tuple[2] else []



            if dst not in destination_connector_pool_cache:

                destination_connector_pool_cache[dst] = _build_destination_connector_target_pool_from_prefix(

                    edge_list=pivot_edges_src,

                    dst=dst,

                    connector_pool_size=cfg.dest_connector_pool_size,

                    active_fallback_pool=active_fallback_target_pool,

                    active_fallback_size=cfg.dest_connector_active_fallback_size,

                )

            destination_connector_target_pool = destination_connector_pool_cache[dst]



            edge_score_cache: Dict[Any, float] = {}

            edge_cache_low_by_tu = defaultdict(set)

            edge_cache_high_by_tu = defaultdict(dict)

            accepted_predicted_out = defaultdict(set)

            compiled_candidate_index_cache: Dict[Any, Any] = {}

            path_state_version = 0

            prev_time = pivot_time

            accepted_records: List[Dict[str, Any]] = []



            for t_cur in data["future_times"]:

                t_cur = int(t_cur)

                if t_cur <= prev_time:

                    continue

                if t_cur > fut_time:

                    break



                _pipeline_step_start = time_lib.perf_counter()

                (

                    candidate_pair_to_best_len,

                    _cached_scored_candidate_occurrences,

                    missing_candidate_occurrences_by_pair,

                    _candidate_build_stats,

                ) = _build_candidate_pairs_fast_with_cache(

                    vertexToLv_dict=L_pivot,

                    dst=dst,

                    reference_len=reference_len,

                    observed_out=data["pivot_observed_out"],

                    accepted_out=accepted_predicted_out,

                    t_cur=t_cur,

                    edge_score_cache=edge_score_cache,

                    edge_cache_low_by_tu=edge_cache_low_by_tu,

                    edge_cache_high_by_tu=edge_cache_high_by_tu,

                    lp_edge_threshold=lp_edge_threshold,

                    oracle_mode="jodie_rr",

                    compiled_candidate_index_cache=compiled_candidate_index_cache,

                    path_state_version=path_state_version,

                    use_edge_score_cache=False,

                    use_candidate_path_index_cache=cfg.use_cache,

                    active_target_pool=destination_connector_target_pool,

                    landmark_target_pool=workload_landmark_pool,

                    landmark_connector_target_pool=landmark_connector_target_pool,

                    landmark_nodes=workload_landmark_pool,

                    root_source=src,

                )



                unique_scored = _score_jodie_candidates_batch(

                    oracle=oracle_query,

                    candidate_pair_to_best_len=candidate_pair_to_best_len,

                    t_cur=t_cur,

                    topk_per_source=cfg.jodie_path_topk_per_source,

                )

                score_by_pair = {(int(u), int(v)): float(p_hat) for u, v, _new_len, p_hat in unique_scored}



                scored_occurrences = []

                for pair_key, occurrences in missing_candidate_occurrences_by_pair.items():

                    p_hat = score_by_pair.get((int(pair_key[0]), int(pair_key[1])))

                    if p_hat is None:

                        continue

                    for lv_path_id, u, v, new_len in occurrences:

                        scored_occurrences.append((int(lv_path_id), int(u), int(v), int(new_len), float(p_hat)))



                accepted_candidates = [cand for cand in scored_occurrences if float(cand[4]) >= float(lp_edge_threshold)]

                if cfg.disable_root_direct_edges:

                    accepted_candidates = [cand for cand in accepted_candidates if not _is_root_direct_candidate(cand, dst)]



                cap_k = cfg.max_accepted_edges_per_lv_path

                if cap_k is not None and int(cap_k) > 0:

                    accepted_by_lv_path = defaultdict(list)

                    for cand in accepted_candidates:

                        accepted_by_lv_path[int(cand[0])].append(cand)

                    limited = []

                    for _lv_path_id, group in accepted_by_lv_path.items():

                        group = sorted(group, key=_accepted_candidate_topk_key)

                        limited.extend(group[: int(cap_k)])

                    accepted_candidates = limited



                pred_stream = []

                predicted_edges_this_time = set()

                unique_accepted_for_update = {}

                for lv_path_id, u, v, new_len, p_hat in accepted_candidates:

                    accepted_predicted_out[int(u)].add(int(v))

                    edge_key = (int(u), int(v), int(t_cur))

                    if edge_key not in unique_accepted_for_update or float(p_hat) > unique_accepted_for_update[edge_key][3]:

                        unique_accepted_for_update[edge_key] = (int(u), int(v), int(new_len), float(p_hat))

                    if edge_key not in predicted_edges_this_time:

                        pred_stream.append((int(u), int(v), int(t_cur), float(p_hat)))

                        predicted_edges_this_time.add(edge_key)

                        accepted_records.append({

                            "u": int(u),

                            "v": int(v),

                            "time": int(t_cur),

                            "score": float(p_hat),

                            "new_len": int(new_len),

                            "lv_path_id": int(lv_path_id),

                        })



                if state_mode == "self_update" and unique_accepted_for_update:

                    accepted_for_update = list(unique_accepted_for_update.values())

                    if cfg.jodie_logic == "jodie_rr":

                        state_update_threshold = 1.0 / max(1, int(cfg.jodie_self_update_top_rank))

                        accepted_for_update = [cand for cand in accepted_for_update if float(cand[3]) >= state_update_threshold]

                    if cfg.jodie_self_update_order == "score_desc":

                        accepted_for_update = sorted(accepted_for_update, key=lambda x: (-float(x[3]), int(x[0]), int(x[1])))

                    else:

                        accepted_for_update = sorted(accepted_for_update, key=lambda x: (int(x[0]), int(x[1]), -float(x[3])))

                    for u_pred, v_pred, _new_len, _p_hat in accepted_for_update:

                        _jodie_update_state_one_edge_inplace(oracle_query, int(u_pred), int(v_pred), int(t_cur))



                prev_time = t_cur

                (

                    shortest_dict,

                    L_dict,

                    actual_dict,

                    dest_paths_dict,

                    best_sp_dict,

                    topk_sp_dict,

                    _candidate_count_dict,

                    _timing_info,

                ) = PredictSP(

                    pred_stream,

                    L_pivot,

                    sp_pivot,

                    actual_pivot,

                    src,

                    data["vertex_list"],

                    (data["pivot_interval"][0], t_cur),

                    dest_paths_pivot,

                    cfg.path_exist_thresholds[0],

                    cfg.shortest_proba_bounds[0],

                    cfg.lk_runs,

                    pivot_time=pivot_time,

                    pivot_baseline_sp=pivot_baseline_sp,

                    return_timing=True,

                    top_k_result_paths=cfg.top_k_result_paths,

                    return_topk=True,

                )



                sp_pivot, L_pivot, actual_pivot, dest_paths_pivot = shortest_dict, L_dict, actual_dict, dest_paths_dict

                if accepted_candidates:

                    path_state_version += 1



                best_tuple = best_sp_dict.get(dst, (math.inf, math.inf, [], [], 0.0, 0.0))

                topk_answer_tuples = list(topk_sp_dict.get(dst, []))

                if not topk_answer_tuples and best_tuple[2]:

                    topk_answer_tuples = [best_tuple]



                d_val, _t_form, path_nodes, _path_edges, _p_sp, _p_exist = best_tuple

                if reference_len is None and d_val != math.inf and path_nodes:

                    reference_len = len(path_nodes) - 1

                answer_tuple = best_tuple

                answer_topk_tuples = topk_answer_tuples



                timing_rows.append(_timing_row_for_step(

                    dataset=data["dataset"],

                    file_stem=data["file_stem"],

                    iteration=iteration,

                    method=method,

                    row=row,

                    pivot_time=pivot_time,

                    t_cur=t_cur,

                    step_total_s=time_lib.perf_counter() - _pipeline_step_start,

                    candidate_pair_to_best_len=candidate_pair_to_best_len,

                    cached_scored_candidate_occurrences=[],

                    missing_candidate_occurrences_by_pair=missing_candidate_occurrences_by_pair,

                    new_scored_candidates=unique_scored,

                    scored_candidates=scored_occurrences,

                    accepted_before_policy=accepted_candidates,

                    accepted_candidates=accepted_candidates,

                    candidate_build_stats=_candidate_build_stats,

                    edge_stats={},

                    candidate_count_dict=_candidate_count_dict,

                    timing_info=_timing_info,

                ))



                if cfg.freeze_answer_at_original_future_time and t_cur >= original_fut_time:

                    break



            prediction_rows.extend(_topk_prediction_rows_for_query(

                dataset=data["dataset"],

                file_stem=data["file_stem"],

                iteration=iteration,

                method=method,

                row=row,

                pivot_time=pivot_time,

                topk_tuples=answer_topk_tuples,

                max_rank=cfg.top_k_result_paths,

            ))

            accepted_edge_rows.extend(_accepted_edge_rows_for_query(

                dataset=data["dataset"],

                file_stem=data["file_stem"],

                iteration=iteration,

                method=method,

                row=row,

                pivot_time=pivot_time,

                accepted_edges=accepted_records,

            ))



            detail_rows.append(compute_query_result(

                dataset=data["dataset"],

                iteration=iteration,

                method=method,

                row=row,

                pivot_time=pivot_time,

                pair_times=data["pair_times"],

                accepted_edges=accepted_records,

                selected_tuple=answer_tuple,

            ))



    _write_method_artifacts(

        cfg=cfg,

        data=data,

        iteration=iteration,

        method=method,

        prediction_rows=prediction_rows,

        timing_rows=timing_rows,

        accepted_edge_rows=accepted_edge_rows,

    )



    return detail_rows





def run_with_progress(label, fn):

    start = time_lib.perf_counter()

    print(f"\n[RUN START] {label}", flush=True)

    rows = fn()

    elapsed = time_lib.perf_counter() - start

    print(

        f"[RUN DONE]  {label} | rows={len(rows)} | seconds={elapsed:.2f}",

        flush=True,

    )

    return rows





def run_all(cfg: PipelineConfig) -> None:

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONFIG] use_cache={cfg.use_cache}", flush=True)

    print(f"[CONFIG] output_dir={cfg.output_dir}", flush=True)



    all_detail_rows: List[Dict[str, Any]] = []



    apply_config_globals(cfg, oracle_mode="static")



    for csv_name, _pivot in DATASETS:

        csv_path = Path(csv_name)

        if not csv_path.exists():

            print(f"[skip] dataset file not found: {csv_path}")

            continue



        print(f"\n=== Dataset: {csv_path.stem} ===")

        data = prepare_dataset(csv_path, cfg)

        if data is None:

            continue



        shared_static = None

        shared_static_embeddings_90 = None

        shared_static_threshold = None

        shared_tgn = None

        shared_tgn_threshold = None

        shared_jodie = None

        shared_jodie_threshold = None



        if cfg.train_lp_once_per_dataset:

            apply_config_globals(cfg, oracle_mode="static")

            emb_dir = cfg.output_dir / "embeddings_static" / f"{data['file_stem']}_shared"

            emb_dir.mkdir(parents=True, exist_ok=True)

            print(f"Training N2VLP-Static once for {data['file_stem']}")

            shared_static, shared_static_embeddings_90, shared_static_threshold, _val, _test = train_static_oracle(

                data=data,

                cfg=cfg,

                iteration=1,

                embeddings_dir=emb_dir,

            )



        if cfg.train_tgn_once_per_dataset:

            apply_config_globals(cfg, oracle_mode="tgn")

            print(f"Training TGN once for {data['file_stem']}")

            shared_tgn, shared_tgn_threshold, _val, _test = train_tgn_oracle(

                data=data,

                cfg=cfg,

                iteration=1,

            )



        if cfg.train_jodie_once_per_dataset:

            apply_config_globals(cfg, oracle_mode="jodie_rr", jodie_state_mode="frozen")

            print(f"Training JODIE once for {data['file_stem']}")

            shared_jodie, shared_jodie_threshold, _val, _test = train_jodie_oracle(

                data=data,

                cfg=cfg,

                iteration=1,

            )



        for iteration in range(1, cfg.num_dataset_iterations + 1):

            print(f"Iteration {iteration}/{cfg.num_dataset_iterations} for {data['file_stem']}")

            tests_df_iter = sample_iteration_tests(data, cfg, iteration)



            if cfg.train_lp_once_per_dataset:

                static_oracle = shared_static

                static_embeddings_90 = shared_static_embeddings_90

                static_threshold = shared_static_threshold

            else:

                apply_config_globals(cfg, oracle_mode="static")

                emb_dir = cfg.output_dir / "embeddings_static" / f"{data['file_stem']}_{iteration}"

                emb_dir.mkdir(parents=True, exist_ok=True)

                static_oracle, static_embeddings_90, static_threshold, _val, _test = train_static_oracle(

                    data=data,

                    cfg=cfg,

                    iteration=iteration,

                    embeddings_dir=emb_dir,

                )



            apply_config_globals(cfg, oracle_mode="static")

            all_detail_rows.extend(run_with_progress(

                f"{data['file_stem']} iter={iteration} method=N2VLP-Static",

                lambda: run_static_or_tgn_pipeline(

                    data=data,

                    cfg=cfg,

                    method="N2VLP-Static",

                    oracle_mode="static",

                    oracle=static_oracle,

                    embeddings_90=static_embeddings_90,

                    lp_edge_threshold=float(static_threshold),

                    tests_df_iter=tests_df_iter,

                    iteration=iteration,

                )

            ))



            if cfg.train_tgn_once_per_dataset:

                tgn_oracle = shared_tgn

                tgn_threshold = shared_tgn_threshold

                _warmup_tgn_oracle(tgn_oracle, data["edges_90"])

            else:

                apply_config_globals(cfg, oracle_mode="tgn")

                tgn_oracle, tgn_threshold, _val, _test = train_tgn_oracle(data=data, cfg=cfg, iteration=iteration)



            apply_config_globals(cfg, oracle_mode="tgn")

            all_detail_rows.extend(run_with_progress(

                f"{data['file_stem']} iter={iteration} method=TGN",

                lambda: run_static_or_tgn_pipeline(

                    data=data,

                    cfg=cfg,

                    method="TGN",

                    oracle_mode="tgn",

                    oracle=tgn_oracle,

                    embeddings_90=None,

                    lp_edge_threshold=float(tgn_threshold),

                    tests_df_iter=tests_df_iter,

                    iteration=iteration,

                )

            ))



            if cfg.train_jodie_once_per_dataset:

                jodie_oracle = shared_jodie

                jodie_threshold = shared_jodie_threshold

            else:

                apply_config_globals(cfg, oracle_mode="jodie_rr", jodie_state_mode="frozen")

                jodie_oracle, jodie_threshold, _val, _test = train_jodie_oracle(data=data, cfg=cfg, iteration=iteration)



            _jodie_warmup_state(jodie_oracle, data["edges_90"])



            all_detail_rows.extend(run_with_progress(

                f"{data['file_stem']} iter={iteration} method=Jodie-Frozen",

                lambda: run_jodie_pipeline(

                    data=data,

                    cfg=cfg,

                    method="Jodie-Frozen",

                    state_mode="frozen",

                    oracle_frozen=jodie_oracle,

                    lp_edge_threshold=float(jodie_threshold),

                    tests_df_iter=tests_df_iter,

                    iteration=iteration,

                )

            ))



            _jodie_warmup_state(jodie_oracle, data["edges_90"])



            all_detail_rows.extend(run_with_progress(

                f"{data['file_stem']} iter={iteration} method=Jodie-Update",

                lambda: run_jodie_pipeline(

                    data=data,

                    cfg=cfg,

                    method="Jodie-Update",

                    state_mode="self_update",

                    oracle_frozen=jodie_oracle,

                    lp_edge_threshold=float(jodie_threshold),

                    tests_df_iter=tests_df_iter,

                    iteration=iteration,

                )

            ))



    if not all_detail_rows:

        raise RuntimeError("No result rows were produced. Check dataset paths and test files.")



    detail_df = pd.DataFrame(all_detail_rows)

    summary_df = summarize_results(detail_df)



    detail_path = cfg.output_dir / "oracle_temporal_results_detail.tsv"

    summary_path = cfg.output_dir / "oracle_temporal_results_summary.tsv"

    latex_path = cfg.output_dir / "oracle_temporal_results_table.tex"



    detail_df.to_csv(detail_path, sep="\t", index=False)

    summary_df.to_csv(summary_path, sep="\t", index=False)

    latex_path.write_text(make_latex_table(summary_df), encoding="utf-8")



    print(f"Saved detail results:      {detail_path}")

    print(f"Saved summary results:     {summary_path}")

    print(f"Saved LaTeX table:        {latex_path}")

    print(summary_df.to_string(index=False))

