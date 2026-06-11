from __future__ import annotations





import copy

import math

import sys

import time as time_lib

from collections import defaultdict

from pathlib import Path

from typing import Dict, Iterable, List, Optional, Tuple



import numpy as np

import pandas as pd










BASE_DIR = Path(__file__).resolve().parents[1]

sys.path.append(str(BASE_DIR / "Helpers"))

sys.path.append(str(BASE_DIR / "FuturePathEstimator"))



import read_Edge_Stream as RES

from path_Prediction_Algorithms import (

    predictBiggestSPDistanceShortestPathAndDistance as PredictSP,

)



TESTS_DIR = BASE_DIR / "Data" / "query_tests"

TOP100_TEST_SUFFIX = ".tsv"



DATASETS = [

    ("Data/Datasets/enron.csv", -1),

    ("Data/Datasets/email-eu.csv", -1),

    ("Data/Datasets/collegemsg.csv", -1),

    ("Data/Datasets/bitcoin.csv", -1),

]




ONLY_DATASET_STEMS = None

MAX_TEST_ROWS_PER_DATASET = None



OUTPUT_BASE_DIR = BASE_DIR / "Results" / "RQ1_candidate_space_oracle"

OUTPUT_DIR = OUTPUT_BASE_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)




EXISTENCE_BOUND = 0.0

SHORTEST_PROBA_BOUND = 0.0

LK_RUNS = 20




USE_DESTINATION_CONNECTOR_TARGETS = True

DEST_CONNECTOR_POOL_SIZE = 50

DEST_CONNECTOR_ACTIVE_FALLBACK_SIZE = 20

DEST_CONNECTOR_SOURCE_BUDGET = 5



USE_PREFIX_SUPPORTED_LANDMARKS = True

WORKLOAD_LANDMARK_POOL_SIZE = 3




LANDMARK_POOL_SIZES_TO_RUN = [1, 3, 5]






RUN_ALL_CANDIDATE_STRESS_PASS = True

STRESS_CANDIDATE_PROBABILITY = 1.0






MAX_STRESS_CANDIDATES_PER_TIMESTAMP = None



LANDMARK_CONNECTOR_POOL_SIZE = 50

LANDMARK_CONNECTOR_WEIGHT = 2.0

LANDMARK_PREFIX_PATH_WEIGHT = 1.0

LANDMARK_REACHABILITY_WEIGHT = 0.25

LANDMARK_DESTINATION_FREQUENCY_WEIGHT = 0.0

LANDMARK_REACHABILITY_MAX_DISTANCE = 3



INCLUDE_CONNECTOR_SOURCES_AS_CANDIDATE_SOURCES = True

INCLUDE_LANDMARK_CONNECTOR_SOURCES = True

INCLUDE_REACHED_LANDMARK_SOURCES = True

INCLUDE_PATH_TARGETS_IN_CONNECTOR_MODE = False








INCLUDE_REACHED_NODES_AS_COMPLETION_SOURCES = True





DISABLE_ROOT_DIRECT_EDGES = False















def _resolve_existing_file(path_str: str) -> Path:

    p = Path(path_str)

    if not p.is_absolute():

        p = BASE_DIR / p

    if p.exists():

        return p



    stem = Path(path_str).stem

    suffix = Path(path_str).suffix

    fallback = BASE_DIR / f"{stem}(1){suffix}"

    if fallback.exists():

        return fallback



    raise FileNotFoundError(f"Missing dataset file: tried {p} and {fallback}")





def _normalize_df(csv_path: Path) -> pd.DataFrame:

    df = pd.read_csv(csv_path, sep=",")

    if {"u", "i", "ts_str"}.issubset(df.columns):

        df = df.rename(columns={"u": "source", "i": "target", "ts_str": "time"})

    elif {"source", "destination", "time"}.issubset(df.columns):

        df = df.rename(columns={"destination": "target"})

    elif {"source", "target", "ts"}.issubset(df.columns) and "time" not in df.columns:

        df = df.rename(columns={"ts": "time"})

    if not {"source", "target", "time"}.issubset(df.columns):

        raise ValueError(f"{csv_path}: needs columns source,target,time")

    df = df[["source", "target", "time"]].copy()

    df["source"] = df["source"].astype(int)

    df["target"] = df["target"].astype(int)

    df["time"] = df["time"].astype(int)

    return df





def _build_full_edge_list(df: pd.DataFrame) -> List[Tuple[int, int, int, float]]:

    edges = [

        (int(u), int(v), int(t), 1.0)

        for u, v, t in df[["source", "target", "time"]].itertuples(index=False)

    ]

    edges.sort(key=lambda x: x[2])

    return edges





def _build_streams(df: pd.DataFrame, pivot_time: int):

    edges = _build_full_edge_list(df)

    vertex_set = set()

    pivot_edges = []

    for u, v, t, p in edges:

        if int(t) <= int(pivot_time):

            vertex_set.add(int(u))

            vertex_set.add(int(v))

            pivot_edges.append((int(u), int(v), int(t), float(p)))



    all_edges = [

        (int(u), int(v), int(t), float(p))

        for u, v, t, p in edges

        if int(u) in vertex_set and int(v) in vertex_set

    ]



    future_times = []

    seen_t = set()

    for _u, _v, t, _p in all_edges:

        if int(t) > int(pivot_time) and int(t) not in seen_t:

            seen_t.add(int(t))

            future_times.append(int(t))



    tmin = int(df["time"].min())

    tmax = int(df["time"].max())

    return (

        sorted(all_edges, key=lambda e: e[2]),

        sorted(pivot_edges, key=lambda e: e[2]),

        sorted(vertex_set),

        sorted(future_times),

        (tmin - 1, tmax + 1),

        (tmin - 1, int(pivot_time)),

    )





def _compute_splits_like_main(df: pd.DataFrame):

    full_edges = _build_full_edge_list(df)

    unique_times = sorted({int(t) for _u, _v, t, _p in full_edges})

    if not unique_times:

        raise ValueError("Dataset has no edges.")



    idx80 = int(0.8 * len(unique_times))

    idx90 = int(0.9 * len(unique_times))

    idx80 = min(max(idx80, 0), len(unique_times) - 1)

    idx90 = min(max(idx90, 0), len(unique_times) - 1)



    t80 = int(unique_times[idx80])

    t90 = int(unique_times[idx90])

    pivot_time = t90



    all_edges, pivot_edges, vertex_list, future_times, global_interval, pivot_interval = _build_streams(

        df, pivot_time

    )



    pivot_observed_out = defaultdict(set)

    for u, v, _t, _p in pivot_edges:

        pivot_observed_out[int(u)].add(int(v))



    return {

        "full_edges": full_edges,

        "t80": t80,

        "t90": t90,

        "pivot_time": pivot_time,

        "all_edges": all_edges,

        "pivot_edges": pivot_edges,

        "vertex_list": vertex_list,

        "future_times": future_times,

        "global_interval": global_interval,

        "pivot_interval": pivot_interval,

        "pivot_observed_out": pivot_observed_out,

    }





def _parse_path_str(x) -> List[int]:

    if pd.isna(x):

        return []

    s = str(x).strip()

    if not s:

        return []

    return [int(z) for z in s.split(",") if str(z).strip()]





def _query_test_candidates(file_stem: str) -> List[Path]:

    aliases = {
        "enron": "enron",
        "email-eu": "email_eu",
        "email_eu": "email_eu",
        "email-eu-core-temporal": "email_eu",
        "email_eu_core_temporal": "email_eu",
        "collegemsg": "collegemsg",
        "CollegeMsg": "collegemsg",
        "college_msg": "collegemsg",
        "bitcoin": "bitcoin",
        "bitcoinotc": "bitcoin",
        "bitcoin_otc": "bitcoin",
    }

    normalized = aliases.get(file_stem, aliases.get(file_stem.lower(), file_stem))

    candidates = [
        TESTS_DIR / f"{normalized}.tsv",
        TESTS_DIR / f"{file_stem}.tsv",
        TESTS_DIR / f"{file_stem}{TOP100_TEST_SUFFIX}",
        BASE_DIR / "new_path_final_Tests" / f"{file_stem}_Path_tests.tsv",
        BASE_DIR / "new_path_final_Tests_top100_temporal_similar" / f"{file_stem}_Path_tests_top100_temporal_similar.tsv",
    ]

    seen = set()
    out = []
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _load_tests(file_stem: str) -> pd.DataFrame:

    tried = _query_test_candidates(file_stem)
    p = None
    for candidate in tried:
        if candidate.exists():
            p = candidate
            break

    if p is None:

        raise FileNotFoundError(
            f"Missing test file for {file_stem}. Tried: "
            + ", ".join(str(x) for x in tried)
        )

    df = pd.read_csv(p, sep="\t")

    required = {"source", "destination", "prev_Path", "future_Path", "time"}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"{p}: missing columns {sorted(missing)}")

    return df





def _pair_to_times(full_edges) -> Dict[Tuple[int, int], List[int]]:

    out = defaultdict(list)

    for u, v, t, _p in full_edges:

        out[(int(u), int(v))].append(int(t))

    return {k: sorted(set(v)) for k, v in out.items()}





def _first_time_strictly_after(times: Iterable[int], lower: int) -> Optional[int]:

    for t in sorted(int(x) for x in times):

        if int(t) > int(lower):

            return int(t)

    return None





def _reconstruct_temporal_edges(path_nodes: List[int], pair_times: Dict[Tuple[int, int], List[int]]):


    edges = []

    last_t = -10**30

    for hop, (u, v) in enumerate(zip(path_nodes[:-1], path_nodes[1:])):

        times = pair_times.get((int(u), int(v)), [])

        if not times:

            return edges, f"missing_stream_edge:{u}->{v}"

        t = _first_time_strictly_after(times, last_t)

        if t is None:

            return edges, f"no_time_respecting_occurrence:{u}->{v}:after:{last_t}"

        edges.append((int(u), int(v), int(t), int(hop)))

        last_t = int(t)

    return edges, "ok"










def _build_global_activity_pool_from_prefix(edge_list, pool_size):

    if pool_size is None or int(pool_size) <= 0:

        return frozenset()

    activity = defaultdict(int)

    for u, v, _t, _p in edge_list:

        activity[int(u)] += 1

        activity[int(v)] += 1

    ranked = sorted(activity.items(), key=lambda item: (-int(item[1]), int(item[0])))

    return frozenset(int(node) for node, _score in ranked[:int(pool_size)])





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

            nodes = _parse_path_str(path_str)

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

    return frozenset(int(x) for *_rest, x in scored[:pool_size])





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










def _extract_nodes_from_path_record_view(path_record):

    if isinstance(path_record, dict):

        return path_record.get("nodes", ())

    if isinstance(path_record, (tuple, list)) and len(path_record) >= 3:

        nodes = path_record[2]

        if isinstance(nodes, (list, tuple)):

            return nodes

    return ()





def _freeze_for_path_cache_key(value):

    if isinstance(value, dict):

        return tuple(sorted((_freeze_for_path_cache_key(k), _freeze_for_path_cache_key(v)) for k, v in value.items()))

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





def _path_record_T(path_record):


    if isinstance(path_record, dict):

        if "T" in path_record:

            return int(path_record["T"])

        edges = path_record.get("edges", [])

        if edges:

            return int(edges[-1][2])

        return -10**30



    if isinstance(path_record, (tuple, list)):

        if len(path_record) >= 2:

            return int(path_record[1])

        return -10**30



    return -10**30





def _compile_candidate_records_for_path(path_record, path_key, dst, reference_len, lv_path_id):

    dst = int(dst)

    nodes = _extract_nodes_from_path_record_view(path_record)

    if not nodes:

        return None



    nodes_tuple = tuple(int(x) for x in nodes)

    if len(nodes_tuple) == 0:

        return None



    nodes_set = frozenset(nodes_tuple)

    label_T = _path_record_T(path_record)

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

        records.append((u, int(lv_path_id), new_len, nodes_set, int(label_T)))



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

):


    dst = int(dst)

    ref_key = -1 if reference_len is None else int(reference_len)

    cache_state = _init_incremental_path_candidate_cache(compiled_candidate_index_cache)

    if cache_state is None:

        cache_state = _init_incremental_path_candidate_cache({})



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

    entries = cache_state["entries"]

    path_ids = cache_state["path_ids"]



    removed_entry_keys = previous_active_keys - active_entry_key_set

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



    cache_state["last_active_entry_keys"] = active_entry_key_set



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

        for u, lv_path_id, new_len, nodes_set, label_T in entry["records"]:

            records_by_source[int(u)].append((int(lv_path_id), int(new_len), nodes_set, int(label_T)))



    candidate_targets = set(target_refcount.keys())

    candidate_targets.add(dst)



    for u in records_by_source:

        records_by_source[u].sort(key=lambda x: (x[1], x[0], x[3]))



    stats = {

        "num_candidate_source_records": sum(len(x) for x in records_by_source.values()),

        "num_candidate_sources": len(records_by_source),

        "num_candidate_targets": len(candidate_targets),

        "num_lv_paths": num_active_entries,

        "candidate_path_index_cache_hit": int(active_entry_key_set == previous_active_keys and added_count == 0),

        "candidate_path_index_cache_size": len(entries),

        "path_candidate_cache_added": added_count,

        "path_candidate_cache_removed": len(removed_entry_keys),

    }

    return records_by_source, frozenset(candidate_targets), stats





def build_candidate_pairs_rewritten(

    vertexToLv_dict,

    dst,

    reference_len,

    observed_out,

    accepted_out,

    compiled_candidate_index_cache=None,

    active_target_pool=None,

    landmark_target_pool=None,

    landmark_connector_target_pool=None,

    landmark_nodes=None,

    root_source=None,

):



    dst = int(dst)



    records_by_source, candidate_targets, index_stats = _get_compiled_candidate_path_index(

        vertexToLv_dict=vertexToLv_dict,

        dst=dst,

        reference_len=reference_len,

        compiled_candidate_index_cache=compiled_candidate_index_cache,

    )



    path_candidate_targets_set = set(candidate_targets)

    connector_target_pool_set = set(int(x) for x in (active_target_pool or ()))

    landmark_nodes_set = set(int(x) for x in (landmark_nodes or ()))

    landmark_target_pool_set = set(int(x) for x in (landmark_target_pool or ()))

    landmark_connector_target_pool_set = set(int(x) for x in (landmark_connector_target_pool or ()))



    connector_mode = bool(

        USE_DESTINATION_CONNECTOR_TARGETS

        and (

            connector_target_pool_set

            or landmark_target_pool_set

            or landmark_connector_target_pool_set

            or landmark_nodes_set

        )

    )



    if connector_mode:

        source_min_lens = []

        for src_u, src_records in records_by_source.items():

            if not src_records:

                continue

            min_new_len = min(int(new_len) for _lv_path_id, new_len, _nodes_set, _label_T in src_records)

            source_min_lens.append((min_new_len, int(src_u)))



        source_min_lens.sort(key=lambda x: (x[0], x[1]))

        connector_source_budget = max(0, int(DEST_CONNECTOR_SOURCE_BUDGET))

        shallow_sources_set = {int(src_u) for _min_len, src_u in source_min_lens[:connector_source_budget]}



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



    candidate_pair_to_best_len: Dict[Tuple[int, int], int] = {}

    occurrences_by_pair: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = defaultdict(list)

    num_candidate_occurrences_total = 0

    completion_only_sources_set = set()



    for u, records in records_by_source.items():

        u = int(u)

        if connector_mode:

            if u in active_target_sources_set:

                candidate_targets_set_for_source = candidate_targets_set

            elif INCLUDE_REACHED_NODES_AS_COMPLETION_SOURCES:





                candidate_targets_set_for_source = {dst}

                completion_only_sources_set.add(u)

            else:

                continue

        else:

            candidate_targets_set_for_source = candidate_targets_set



        blocked = set(int(x) for x in observed_out.get(u, ()))

        blocked.update(int(x) for x in accepted_out.get(u, ()))

        blocked.add(u)



        for lv_path_id, new_len, nodes_set, label_T in records:

            lv_path_id = int(lv_path_id)

            new_len = int(new_len)

            label_T = int(label_T)



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





            if DISABLE_ROOT_DIRECT_EDGES and (

                int(new_len) == 1

                or (root_source is not None and int(u) == int(root_source) and int(new_len) <= 1)

            ):

                valid_targets.discard(dst)



            for v in valid_targets:

                v = int(v)

                pair_key = (u, v)

                old_len = candidate_pair_to_best_len.get(pair_key)

                if old_len is None or new_len < old_len:

                    candidate_pair_to_best_len[pair_key] = new_len

                occurrences_by_pair[pair_key].append((lv_path_id, u, v, new_len, label_T))

                num_candidate_occurrences_total += 1



    stats = {

        "num_candidate_source_records": index_stats.get("num_candidate_source_records", 0),

        "num_candidate_sources": index_stats.get("num_candidate_sources", 0),

        "num_candidate_targets_from_paths": len(path_candidate_targets_set),

        "num_destination_connector_targets": len(connector_target_pool_set),

        "num_landmark_targets": len(landmark_nodes_set),

        "num_landmark_connector_targets": len(landmark_connector_target_pool_set),

        "num_active_target_sources": len(active_target_sources_set),

        "num_shallow_candidate_sources": len(shallow_sources_set),

        "num_connector_candidate_sources": len(connector_sources_set),

        "include_reached_nodes_as_completion_sources": int(INCLUDE_REACHED_NODES_AS_COMPLETION_SOURCES),

        "num_completion_only_candidate_sources": len(completion_only_sources_set),

        "num_candidate_targets_total": len(candidate_targets_set),

        "num_candidate_pairs_total": num_candidate_occurrences_total,

        "num_unique_candidate_pairs": len(candidate_pair_to_best_len),

        "num_lv_paths": index_stats.get("num_lv_paths", 0),

        "disable_root_direct_edges": int(DISABLE_ROOT_DIRECT_EDGES),

    }

    return candidate_pair_to_best_len, occurrences_by_pair, stats





def _pair_is_temporally_feasible(pair, occurrences_by_pair, edge_time):



    edge_time = int(edge_time)

    for occ in occurrences_by_pair.get(pair, []):


        if len(occ) >= 5:

            label_T = int(occ[4])

            if label_T < edge_time:

                return True

        else:


            return True

    return False










def _fresh_query_state(base_state_tuple):

    return copy.deepcopy(base_state_tuple)





def _call_predictsp_update(

    pred_stream,

    L_pivot,

    sp_pivot,

    actual_pivot,

    src,

    vertex_list,

    pivot_interval,

    t_update,

    dest_paths_pivot,

    pivot_time,

    pivot_baseline_sp,

):

    try:

        (

            shortest_dict,

            L_dict,

            actual_dict,

            dest_paths_dict,

            best_sp_dict,

            candidate_count_dict,

            _timing,

        ) = PredictSP(

            pred_stream,

            L_pivot,

            sp_pivot,

            actual_pivot,

            int(src),

            vertex_list,

            (pivot_interval[0], int(t_update)),

            dest_paths_pivot,

            EXISTENCE_BOUND,

            SHORTEST_PROBA_BOUND,

            LK_RUNS,

            pivot_time=int(pivot_time),

            pivot_baseline_sp=pivot_baseline_sp,

            return_timing=True,

        )

    except TypeError:

        (

            shortest_dict,

            L_dict,

            actual_dict,

            dest_paths_dict,

            best_sp_dict,

            candidate_count_dict,

        ) = PredictSP(

            pred_stream,

            L_pivot,

            sp_pivot,

            actual_pivot,

            int(src),

            vertex_list,

            (pivot_interval[0], int(t_update)),

            dest_paths_pivot,

            EXISTENCE_BOUND,

            SHORTEST_PROBA_BOUND,

            LK_RUNS,

            pivot_time=int(pivot_time),

            pivot_baseline_sp=pivot_baseline_sp,

        )

    return shortest_dict, L_dict, actual_dict, dest_paths_dict, best_sp_dict, candidate_count_dict














def _extract_record_nodes_edges_T_p(item):



    if isinstance(item, dict):

        nodes = list(item.get("nodes", []))

        edges = list(item.get("edges", []))

        T = item.get("T", None)

        p = item.get("p", item.get("prob", None))

        return nodes, edges, T, p



    if isinstance(item, (tuple, list)) and len(item) >= 5:

        _d, T, nodes, edges, p = item[:5]

        return list(nodes), list(edges), T, p



    return [], [], None, None





def _edge_sig_from_edges(edges):

    sig = []

    for e in edges:

        if isinstance(e, (tuple, list)) and len(e) >= 3:

            sig.append((int(e[0]), int(e[1]), int(e[2])))

    return sig





def _is_subsequence(needle, haystack):

    if not needle:

        return True

    j = 0

    for x in haystack:

        if x == needle[j]:

            j += 1

            if j == len(needle):

                return True

    return False





def _find_ground_truth_path_in_final_state(

    L_dict,

    destination_paths_dict,

    dst,

    future_nodes,

    future_only_edges,

    query_time,

):



    dst = int(dst)

    query_time = int(query_time)

    gt_nodes = tuple(int(x) for x in future_nodes)

    gt_future_sig = [(int(u), int(v), int(t)) for (u, v, t, _hop) in future_only_edges]



    candidates = []

    for rec in L_dict.get(dst, []):

        candidates.append(("L_dst", rec))

    for rec in destination_paths_dict.get(dst, []):

        candidates.append(("destination_paths", rec))



    nodes_only_match = None

    exact_match = None



    for source_name, rec in candidates:

        nodes, edges, T, p = _extract_record_nodes_edges_T_p(rec)

        if not nodes:

            continue

        if tuple(int(x) for x in nodes) != gt_nodes:

            continue

        if T is not None and int(T) > query_time:

            continue



        edge_sig = _edge_sig_from_edges(edges)

        info = {

            "state_source": source_name,

            "nodes": ",".join(map(str, nodes)),

            "T": int(T) if T is not None else None,

            "p": float(p) if p is not None else None,

            "edge_sig": "|".join(f"{u},{v},{t}" for u, v, t in edge_sig),

        }



        if nodes_only_match is None:

            nodes_only_match = dict(info)



        if _is_subsequence(gt_future_sig, edge_sig):

            exact_match = dict(info)

            break



    return {

        "gt_path_present_final_nodes_only": int(nodes_only_match is not None),

        "gt_path_present_final_exact_future_edges": int(exact_match is not None),

        "gt_path_final_match_source": (exact_match or nodes_only_match or {}).get("state_source"),

        "gt_path_final_match_T": (exact_match or nodes_only_match or {}).get("T"),

        "gt_path_final_match_p": (exact_match or nodes_only_match or {}).get("p"),

        "gt_path_final_match_edges": (exact_match or nodes_only_match or {}).get("edge_sig"),

    }









def _safe_mean(df: pd.DataFrame, col: str):

    if df is None or len(df) == 0 or col not in df.columns:

        return None

    return df[col].mean(skipna=True)





def _safe_sum(df: pd.DataFrame, col: str):

    if df is None or len(df) == 0 or col not in df.columns:

        return None

    return df[col].sum(skipna=True)





def _safe_max(df: pd.DataFrame, col: str):

    if df is None or len(df) == 0 or col not in df.columns:

        return None

    return df[col].max(skipna=True)





def _empty_stress_summary():

    return {

        "stress_enabled": int(RUN_ALL_CANDIDATE_STRESS_PASS),

        "stress_num_rollout_timestamps": 0,

        "stress_num_predictsp_calls": 0,

        "stress_candidate_generation_s": 0.0,

        "stress_predictsp_s": 0.0,

        "stress_total_s": 0.0,

        "stress_sum_unique_candidate_pairs": 0,

        "stress_mean_unique_candidate_pairs": 0.0,

        "stress_max_unique_candidate_pairs": 0,

        "stress_sum_candidate_occurrences": 0,

        "stress_mean_candidate_occurrences": 0.0,

        "stress_max_candidate_occurrences": 0,

        "stress_sum_pred_stream_edges": 0,

        "stress_mean_pred_stream_edges": 0.0,

        "stress_max_pred_stream_edges": 0,

        "stress_candidate_cap_hit": 0,

        "stress_max_candidates_per_timestamp": -1 if MAX_STRESS_CANDIDATES_PER_TIMESTAMP is None else int(MAX_STRESS_CANDIDATES_PER_TIMESTAMP),

    }





def _summarize_stress_rows(stress_rows: List[Dict]):

    if not stress_rows:

        return _empty_stress_summary()



    unique_counts = [int(r.get("num_unique_candidate_pairs", 0)) for r in stress_rows]

    occurrence_counts = [int(r.get("num_candidate_occurrences", 0)) for r in stress_rows]

    stream_counts = [int(r.get("num_pred_stream_edges", 0)) for r in stress_rows]

    cand_s = [float(r.get("candidate_generation_s", 0.0)) for r in stress_rows]

    pred_s = [float(r.get("predictsp_s", 0.0)) for r in stress_rows]



    return {

        "stress_enabled": int(RUN_ALL_CANDIDATE_STRESS_PASS),

        "stress_num_rollout_timestamps": int(len(stress_rows)),

        "stress_num_predictsp_calls": int(sum(1 for r in stress_rows if int(r.get("num_pred_stream_edges", 0)) > 0)),

        "stress_candidate_generation_s": float(sum(cand_s)),

        "stress_predictsp_s": float(sum(pred_s)),

        "stress_total_s": float(sum(cand_s) + sum(pred_s)),

        "stress_sum_unique_candidate_pairs": int(sum(unique_counts)),

        "stress_mean_unique_candidate_pairs": float(np.mean(unique_counts)) if unique_counts else 0.0,

        "stress_max_unique_candidate_pairs": int(max(unique_counts)) if unique_counts else 0,

        "stress_sum_candidate_occurrences": int(sum(occurrence_counts)),

        "stress_mean_candidate_occurrences": float(np.mean(occurrence_counts)) if occurrence_counts else 0.0,

        "stress_max_candidate_occurrences": int(max(occurrence_counts)) if occurrence_counts else 0,

        "stress_sum_pred_stream_edges": int(sum(stream_counts)),

        "stress_mean_pred_stream_edges": float(np.mean(stream_counts)) if stream_counts else 0.0,

        "stress_max_pred_stream_edges": int(max(stream_counts)) if stream_counts else 0,

        "stress_candidate_cap_hit": int(any(int(r.get("candidate_cap_hit", 0)) for r in stress_rows)),

        "stress_max_candidates_per_timestamp": -1 if MAX_STRESS_CANDIDATES_PER_TIMESTAMP is None else int(MAX_STRESS_CANDIDATES_PER_TIMESTAMP),

    }





def _run_all_candidate_stress_pass(

    row,

    dataset_stem,

    split_info,

    base_state_by_source,

    active_fallback_target_pool,

    workload_landmark_pool,

    landmark_connector_target_pool,

    reference_len,

):



    src = int(row.source)

    dst = int(row.destination)

    query_time = int(row.time)

    pivot_time = int(split_info["pivot_time"])



    stress_rows = []

    if not RUN_ALL_CANDIDATE_STRESS_PASS or src not in base_state_by_source:

        return stress_rows, _summarize_stress_rows(stress_rows)



    (

        sp_state,

        L_state,

        actual_state,

        dest_paths_state,

        pivot_baseline_sp,

    ) = _fresh_query_state(base_state_by_source[src])



    destination_connector_target_pool = _build_destination_connector_target_pool_from_prefix(

        edge_list=split_info["pivot_edges"],

        dst=dst,

        connector_pool_size=DEST_CONNECTOR_POOL_SIZE,

        active_fallback_pool=active_fallback_target_pool,

        active_fallback_size=DEST_CONNECTOR_ACTIVE_FALLBACK_SIZE,

    )



    stress_accepted_out = defaultdict(set)

    stress_candidate_index_cache = {}



    for t_cur in split_info["future_times"]:

        t_cur = int(t_cur)

        if t_cur <= pivot_time:

            continue

        if t_cur > query_time:

            break



        cand_start = time_lib.perf_counter()

        candidates, occurrences_by_pair, stats = build_candidate_pairs_rewritten(

            vertexToLv_dict=L_state,

            dst=dst,

            reference_len=reference_len,

            observed_out=split_info["pivot_observed_out"],

            accepted_out=stress_accepted_out,

            compiled_candidate_index_cache=stress_candidate_index_cache,

            active_target_pool=destination_connector_target_pool,

            landmark_target_pool=workload_landmark_pool,

            landmark_connector_target_pool=landmark_connector_target_pool,

            landmark_nodes=workload_landmark_pool,

            root_source=src,

        )

        cand_s = time_lib.perf_counter() - cand_start



        candidate_items = sorted(candidates.items(), key=lambda item: (int(item[0][0]), int(item[0][1])))

        cap_hit = 0

        if MAX_STRESS_CANDIDATES_PER_TIMESTAMP is not None:

            cap = int(MAX_STRESS_CANDIDATES_PER_TIMESTAMP)

            if cap >= 0 and len(candidate_items) > cap:

                candidate_items = candidate_items[:cap]

                cap_hit = 1



        pred_stream = [

            (int(u), int(v), int(t_cur), float(STRESS_CANDIDATE_PROBABILITY))

            for (u, v), _new_len in candidate_items

        ]



        predictsp_s = 0.0

        if pred_stream:

            pred_start = time_lib.perf_counter()

            (

                sp_state,

                L_state,

                actual_state,

                dest_paths_state,

                _best_sp,

                _candidate_counts,

            ) = _call_predictsp_update(

                pred_stream=pred_stream,

                L_pivot=L_state,

                sp_pivot=sp_state,

                actual_pivot=actual_state,

                src=src,

                vertex_list=split_info["vertex_list"],

                pivot_interval=split_info["pivot_interval"],

                t_update=t_cur,

                dest_paths_pivot=dest_paths_state,

                pivot_time=pivot_time,

                pivot_baseline_sp=pivot_baseline_sp,

            )

            predictsp_s = time_lib.perf_counter() - pred_start



            for u, v, _t, _p in pred_stream:

                stress_accepted_out[int(u)].add(int(v))



            stress_candidate_index_cache.clear()



        stress_rows.append({

            "dataset": dataset_stem,

            "source": src,

            "destination": dst,

            "query_time": query_time,

            "pivot_time": pivot_time,

            "landmark_pool_size": int(WORKLOAD_LANDMARK_POOL_SIZE),

            "t_cur": t_cur,

            "candidate_generation_s": float(cand_s),

            "predictsp_s": float(predictsp_s),

            "stress_step_total_s": float(cand_s + predictsp_s),

            "num_unique_candidate_pairs": int(len(candidates)),

            "num_candidate_occurrences": int(stats.get("num_candidate_pairs_total", 0)),

            "num_pred_stream_edges": int(len(pred_stream)),

            "candidate_cap_hit": int(cap_hit),

            "num_candidate_sources": int(stats.get("num_candidate_sources", 0)),

            "num_candidate_source_records": int(stats.get("num_candidate_source_records", 0)),

            "num_candidate_targets_from_paths": int(stats.get("num_candidate_targets_from_paths", 0)),

            "num_destination_connector_targets": int(stats.get("num_destination_connector_targets", 0)),

            "num_landmark_targets": int(stats.get("num_landmark_targets", 0)),

            "num_landmark_connector_targets": int(stats.get("num_landmark_connector_targets", 0)),

            "num_active_target_sources": int(stats.get("num_active_target_sources", 0)),

            "num_shallow_candidate_sources": int(stats.get("num_shallow_candidate_sources", 0)),

            "num_connector_candidate_sources": int(stats.get("num_connector_candidate_sources", 0)),

            "num_completion_only_candidate_sources": int(stats.get("num_completion_only_candidate_sources", 0)),

            "num_candidate_targets_total": int(stats.get("num_candidate_targets_total", 0)),

            "num_lv_paths": int(stats.get("num_lv_paths", 0)),

        })



    return stress_rows, _summarize_stress_rows(stress_rows)








def audit_one_query(

    row,

    dataset_stem,

    split_info,

    pair_times,

    base_state_by_source,

    active_fallback_target_pool,

    workload_landmark_pool,

    landmark_connector_target_pool,

):

    src = int(row.source)

    dst = int(row.destination)

    query_time = int(row.time)

    pivot_time = int(split_info["pivot_time"])



    prev_nodes = _parse_path_str(row.prev_Path)

    future_nodes = _parse_path_str(row.future_Path)



    temporal_edges, reconstruction_status = _reconstruct_temporal_edges(future_nodes, pair_times)

    future_only_edges = [

        (u, v, t, hop)

        for (u, v, t, hop) in temporal_edges

        if int(t) > pivot_time and int(t) <= query_time

    ]



    q_base = {

        "dataset": dataset_stem,

        "source": src,

        "destination": dst,

        "query_time": query_time,

        "pivot_time": pivot_time,

        "prev_Path": ",".join(map(str, prev_nodes)),

        "future_Path": ",".join(map(str, future_nodes)),

        "temporal_reconstruction_status": reconstruction_status,

        "future_Path_temporal_edges": "|".join(f"{u},{v},{t},{hop}" for u, v, t, hop in temporal_edges),

        "future_only_edges": "|".join(f"{u},{v},{t},{hop}" for u, v, t, hop in future_only_edges),

        "num_future_only_edges": len(future_only_edges),

        "landmark_pool_size": int(WORKLOAD_LANDMARK_POOL_SIZE),

        "disable_root_direct_edges": int(DISABLE_ROOT_DIRECT_EDGES),

    }



    edge_rows = []



    if reconstruction_status != "ok" or src not in base_state_by_source:

        q = dict(q_base)

        q.update({

            "failure_reason": reconstruction_status if reconstruction_status != "ok" else "source_not_in_base_state",

            "edge_recall_any_by_query": 0.0,

            "edge_recall_on_time_any": 0.0,

            "edge_recall_ordered_oracle": 0.0,

            "ordered_prefix_recovered": 0,

            "ordered_prefix_recovery_rate": 0.0,

            "path_recoverable_ordered_oracle": 0,

            "first_missing_ordered_edge_index": 0 if future_only_edges else -1,

            "num_edges_found_any_by_query": 0,

            "num_edges_found_on_time_any": 0,

            "num_edges_found_ordered_oracle": 0,

            "gt_path_present_final_nodes_only": 0,

            "gt_path_present_final_exact_future_edges": 0,

            "gt_path_final_match_source": None,

            "gt_path_final_match_T": None,

            "gt_path_final_match_p": None,

            "gt_path_final_match_edges": None,

            "edge_first_ordered_ntmse": None,

            "edge_first_ordered_mae": None,

            "edge_first_ordered_mean_lead_time": None,

            "edge_first_ordered_mean_normalized_lead_time": None,

            "edge_first_on_time_ntmse": None,

            "edge_first_on_time_mae": None,

            "edge_first_on_time_mean_lead_time": None,

            "edge_first_on_time_mean_normalized_lead_time": None,

            "num_unique_candidate_pairs_by_query": 0,

            "num_unique_candidate_pairs_actual_future_any": 0,

            "num_unique_candidate_pairs_actual_future_on_time": 0,

            "num_unique_candidate_pairs_true_path_any": 0,

            "num_unique_candidate_pairs_true_path_on_time": 0,

            "edge_candidate_precision_actual_future_any": None,

            "edge_candidate_precision_actual_future_on_time": None,

            "edge_candidate_precision_path_any": None,

            "edge_candidate_precision_path_on_time": None,

        })

        q.update(_empty_stress_summary())

        return edge_rows, q, []



    if not future_only_edges:

        q = dict(q_base)

        q.update({

            "failure_reason": "no_future_only_edges",

            "edge_recall_any_by_query": None,

            "edge_recall_on_time_any": None,

            "edge_recall_ordered_oracle": None,

            "ordered_prefix_recovered": 0,

            "ordered_prefix_recovery_rate": None,

            "path_recoverable_ordered_oracle": 1,

            "first_missing_ordered_edge_index": -1,

            "num_edges_found_any_by_query": 0,

            "num_edges_found_on_time_any": 0,

            "num_edges_found_ordered_oracle": 0,

            "gt_path_present_final_nodes_only": 0,

            "gt_path_present_final_exact_future_edges": 0,

            "gt_path_final_match_source": None,

            "gt_path_final_match_T": None,

            "gt_path_final_match_p": None,

            "gt_path_final_match_edges": None,

            "edge_first_ordered_ntmse": None,

            "edge_first_ordered_mae": None,

            "edge_first_ordered_mean_lead_time": None,

            "edge_first_ordered_mean_normalized_lead_time": None,

            "edge_first_on_time_ntmse": None,

            "edge_first_on_time_mae": None,

            "edge_first_on_time_mean_lead_time": None,

            "edge_first_on_time_mean_normalized_lead_time": None,

            "num_unique_candidate_pairs_by_query": 0,

            "num_unique_candidate_pairs_actual_future_any": 0,

            "num_unique_candidate_pairs_actual_future_on_time": 0,

            "num_unique_candidate_pairs_true_path_any": 0,

            "num_unique_candidate_pairs_true_path_on_time": 0,

            "edge_candidate_precision_actual_future_any": None,

            "edge_candidate_precision_actual_future_on_time": None,

            "edge_candidate_precision_path_any": None,

            "edge_candidate_precision_path_on_time": None,

        })

        q.update(_empty_stress_summary())

        return edge_rows, q, []



    (

        sp_pivot,

        L_pivot,

        actual_pivot,

        dest_paths_pivot,

        pivot_baseline_sp,

    ) = _fresh_query_state(base_state_by_source[src])



    pivot_entry = pivot_baseline_sp.get(dst, (math.inf, [], 0, 0.0))

    pivot_nodes = pivot_entry[1]

    reference_len = len(pivot_nodes) - 1 if pivot_nodes else None



    destination_connector_target_pool = _build_destination_connector_target_pool_from_prefix(

        edge_list=split_info["pivot_edges"],

        dst=dst,

        connector_pool_size=DEST_CONNECTOR_POOL_SIZE,

        active_fallback_pool=active_fallback_target_pool,

        active_fallback_size=DEST_CONNECTOR_ACTIVE_FALLBACK_SIZE,

    )



    accepted_predicted_out = defaultdict(set)

    compiled_candidate_index_cache = {}

    candidate_times_by_edge = {i: [] for i in range(len(future_only_edges))}

    first_candidate_time_any = {i: None for i in range(len(future_only_edges))}

    first_candidate_time_on_time = {i: None for i in range(len(future_only_edges))}

    first_candidate_time_ordered = {i: None for i in range(len(future_only_edges))}

    oracle_accept_time = {i: None for i in range(len(future_only_edges))}

    candidate_new_len_first = {i: None for i in range(len(future_only_edges))}

    num_candidates_when_first_seen = {i: None for i in range(len(future_only_edges))}

    next_true_idx = 0

    pending_true_edge_seen = False

    pending_true_edge_first_seen_time = None





    candidate_first_time_by_pair = {}

    candidate_occurrence_count_by_pair = defaultdict(int)



    for t_cur in split_info["future_times"]:

        t_cur = int(t_cur)

        if t_cur <= pivot_time:

            continue

        if t_cur > query_time:

            break



        candidates, occurrences_by_pair, stats = build_candidate_pairs_rewritten(

            vertexToLv_dict=L_pivot,

            dst=dst,

            reference_len=reference_len,

            observed_out=split_info["pivot_observed_out"],

            accepted_out=accepted_predicted_out,

            compiled_candidate_index_cache=compiled_candidate_index_cache,

            active_target_pool=destination_connector_target_pool,

            landmark_target_pool=workload_landmark_pool,

            landmark_connector_target_pool=landmark_connector_target_pool,

            landmark_nodes=workload_landmark_pool,

            root_source=src,

        )



        for pair_key in candidates.keys():

            pair_key = (int(pair_key[0]), int(pair_key[1]))

            if pair_key not in candidate_first_time_by_pair:

                candidate_first_time_by_pair[pair_key] = int(t_cur)

            candidate_occurrence_count_by_pair[pair_key] += 1



        for i, (u, v, actual_t, _hop) in enumerate(future_only_edges):

            pair_i = (int(u), int(v))

            if (

                pair_i in candidates

                and _pair_is_temporally_feasible(pair_i, occurrences_by_pair, actual_t)

            ):

                candidate_times_by_edge[i].append(t_cur)

                if first_candidate_time_any[i] is None:

                    first_candidate_time_any[i] = t_cur

                if t_cur <= int(actual_t) and first_candidate_time_on_time[i] is None:

                    first_candidate_time_on_time[i] = t_cur



        if next_true_idx < len(future_only_edges):

            u_next, v_next, actual_t_next, _hop_next = future_only_edges[next_true_idx]

            pair_next = (int(u_next), int(v_next))

            actual_t_next = int(actual_t_next)






            if (

                not pending_true_edge_seen

                and pair_next in candidates

                and t_cur <= actual_t_next

                and _pair_is_temporally_feasible(pair_next, occurrences_by_pair, actual_t_next)

            ):

                pending_true_edge_seen = True

                pending_true_edge_first_seen_time = t_cur

                first_candidate_time_ordered[next_true_idx] = t_cur

                candidate_new_len_first[next_true_idx] = candidates[pair_next]

                num_candidates_when_first_seen[next_true_idx] = len(candidates)





            if pending_true_edge_seen and t_cur >= actual_t_next:

                pred_stream = [(int(u_next), int(v_next), actual_t_next, 1.0)]

                accepted_predicted_out[int(u_next)].add(int(v_next))



                (

                    sp_pivot,

                    L_pivot,

                    actual_pivot,

                    dest_paths_pivot,

                    _best_sp,

                    _candidate_counts,

                ) = _call_predictsp_update(

                    pred_stream=pred_stream,

                    L_pivot=L_pivot,

                    sp_pivot=sp_pivot,

                    actual_pivot=actual_pivot,

                    src=src,

                    vertex_list=split_info["vertex_list"],

                    pivot_interval=split_info["pivot_interval"],

                    t_update=actual_t_next,

                    dest_paths_pivot=dest_paths_pivot,

                    pivot_time=pivot_time,

                    pivot_baseline_sp=pivot_baseline_sp,

                )

                oracle_accept_time[next_true_idx] = actual_t_next

                compiled_candidate_index_cache.clear()

                next_true_idx += 1

                pending_true_edge_seen = False

                pending_true_edge_first_seen_time = None



    stress_rows, stress_summary = _run_all_candidate_stress_pass(

        row=row,

        dataset_stem=dataset_stem,

        split_info=split_info,

        base_state_by_source=base_state_by_source,

        active_fallback_target_pool=active_fallback_target_pool,

        workload_landmark_pool=workload_landmark_pool,

        landmark_connector_target_pool=landmark_connector_target_pool,

        reference_len=reference_len,

    )



    final_gt_path_info = _find_ground_truth_path_in_final_state(

        L_dict=L_pivot,

        destination_paths_dict=dest_paths_pivot,

        dst=dst,

        future_nodes=future_nodes,

        future_only_edges=future_only_edges,

        query_time=query_time,

    )



    first_missing_idx = -1

    for i, (u, v, actual_t, hop) in enumerate(future_only_edges):

        times_any = sorted(set(candidate_times_by_edge[i]))

        valid_times_on_time = [t for t in times_any if int(t) <= int(actual_t)]

        closest_on_time = (

            min(valid_times_on_time, key=lambda t: (abs(int(t) - int(actual_t)), int(t)))

            if valid_times_on_time else None

        )



        found_any = first_candidate_time_any[i] is not None

        found_on_time = first_candidate_time_on_time[i] is not None

        found_ordered = first_candidate_time_ordered[i] is not None

        if first_missing_idx == -1 and not found_ordered:

            first_missing_idx = i






        query_time_denom = max(1, int(query_time) - int(pivot_time))



        edge_first_ordered_time_error = None

        edge_first_ordered_abs_time_error = None

        edge_first_ordered_ntse = None

        edge_first_ordered_lead_time = None

        edge_first_ordered_normalized_lead_time = None



        if found_ordered:

            t_hat_ordered = int(first_candidate_time_ordered[i])

            t_true = int(actual_t)

            edge_first_ordered_time_error = t_hat_ordered - t_true

            edge_first_ordered_abs_time_error = abs(edge_first_ordered_time_error)

            edge_first_ordered_ntse = (edge_first_ordered_time_error / query_time_denom) ** 2

            edge_first_ordered_lead_time = t_true - t_hat_ordered

            edge_first_ordered_normalized_lead_time = edge_first_ordered_lead_time / query_time_denom



        edge_first_on_time_time_error = None

        edge_first_on_time_abs_time_error = None

        edge_first_on_time_ntse = None

        edge_first_on_time_lead_time = None

        edge_first_on_time_normalized_lead_time = None



        if found_on_time:

            t_hat_on_time = int(first_candidate_time_on_time[i])

            t_true = int(actual_t)

            edge_first_on_time_time_error = t_hat_on_time - t_true

            edge_first_on_time_abs_time_error = abs(edge_first_on_time_time_error)

            edge_first_on_time_ntse = (edge_first_on_time_time_error / query_time_denom) ** 2

            edge_first_on_time_lead_time = t_true - t_hat_on_time

            edge_first_on_time_normalized_lead_time = edge_first_on_time_lead_time / query_time_denom



        edge_rows.append({

            **q_base,

            "future_edge_index": i,

            "future_edge_hop_in_path": hop,

            "edge_u": int(u),

            "edge_v": int(v),

            "actual_edge_time": int(actual_t),

            "candidate_found_any_by_query": int(found_any),

            "candidate_found_before_or_at_actual_time": int(found_on_time),

            "candidate_found_ordered_oracle": int(found_ordered),

            "first_candidate_time_any": first_candidate_time_any[i],

            "first_candidate_time_before_or_at_actual": first_candidate_time_on_time[i],

            "first_candidate_time_ordered_oracle": first_candidate_time_ordered[i],

            "closest_candidate_time_before_or_at_actual": closest_on_time,

            "all_candidate_times_by_query": "|".join(map(str, times_any)),

            "candidate_new_len_first_ordered": candidate_new_len_first[i],

            "num_candidates_when_first_seen_ordered": num_candidates_when_first_seen[i],

            "oracle_accept_time": oracle_accept_time[i],

            "edge_first_ordered_time_error": edge_first_ordered_time_error,

            "edge_first_ordered_abs_time_error": edge_first_ordered_abs_time_error,

            "edge_first_ordered_ntse": edge_first_ordered_ntse,

            "edge_first_ordered_lead_time": edge_first_ordered_lead_time,

            "edge_first_ordered_normalized_lead_time": edge_first_ordered_normalized_lead_time,

            "edge_first_on_time_time_error": edge_first_on_time_time_error,

            "edge_first_on_time_abs_time_error": edge_first_on_time_abs_time_error,

            "edge_first_on_time_ntse": edge_first_on_time_ntse,

            "edge_first_on_time_lead_time": edge_first_on_time_lead_time,

            "edge_first_on_time_normalized_lead_time": edge_first_on_time_normalized_lead_time,

        })



    n = len(future_only_edges)

    found_any_count = sum(r["candidate_found_any_by_query"] for r in edge_rows)

    found_on_time_count = sum(r["candidate_found_before_or_at_actual_time"] for r in edge_rows)

    found_ordered_count = sum(r["candidate_found_ordered_oracle"] for r in edge_rows)

    ordered_prefix = 0

    for r in sorted(edge_rows, key=lambda x: x["future_edge_index"]):

        if int(r["candidate_found_ordered_oracle"]) == 1:

            ordered_prefix += 1

        else:

            break



    def _edge_metric_mean(key):

        vals = [r.get(key) for r in edge_rows if r.get(key) is not None]

        return float(np.mean(vals)) if vals else None



    candidate_pairs_by_query = set(candidate_first_time_by_pair.keys())

    num_unique_candidate_pairs_by_query = len(candidate_pairs_by_query)



    actual_future_pair_times = defaultdict(list)

    for u_all, v_all, t_all, _p_all in split_info["all_edges"]:

        t_all = int(t_all)

        if int(pivot_time) < t_all <= int(query_time):

            actual_future_pair_times[(int(u_all), int(v_all))].append(t_all)



    true_path_future_pair_times = defaultdict(list)

    for u_gt, v_gt, t_gt, _hop_gt in future_only_edges:

        true_path_future_pair_times[(int(u_gt), int(v_gt))].append(int(t_gt))



    actual_future_any_pairs = {

        pair for pair in candidate_pairs_by_query

        if pair in actual_future_pair_times

    }

    actual_future_on_time_pairs = {

        pair for pair in candidate_pairs_by_query

        if any(

            int(candidate_first_time_by_pair[pair]) <= int(t_actual)

            for t_actual in actual_future_pair_times.get(pair, [])

        )

    }

    true_path_any_pairs = {

        pair for pair in candidate_pairs_by_query

        if pair in true_path_future_pair_times

    }

    true_path_on_time_pairs = {

        pair for pair in candidate_pairs_by_query

        if any(

            int(candidate_first_time_by_pair[pair]) <= int(t_actual)

            for t_actual in true_path_future_pair_times.get(pair, [])

        )

    }



    def _precision(num_correct):

        if num_unique_candidate_pairs_by_query == 0:

            return None

        return float(num_correct) / float(num_unique_candidate_pairs_by_query)



    q = dict(q_base)

    q.update({

        "failure_reason": "ok" if ordered_prefix == n else "candidate_missing_or_not_unlocked",

        "edge_recall_any_by_query": found_any_count / n if n else None,

        "edge_recall_on_time_any": found_on_time_count / n if n else None,

        "edge_recall_ordered_oracle": found_ordered_count / n if n else None,

        "ordered_prefix_recovered": ordered_prefix,

        "ordered_prefix_recovery_rate": ordered_prefix / n if n else None,

        "path_recoverable_ordered_oracle": int(ordered_prefix == n),

        "first_missing_ordered_edge_index": first_missing_idx,

        "num_edges_found_any_by_query": found_any_count,

        "num_edges_found_on_time_any": found_on_time_count,

        "num_edges_found_ordered_oracle": found_ordered_count,

        "num_unique_candidate_pairs_by_query": num_unique_candidate_pairs_by_query,

        "num_unique_candidate_pairs_actual_future_any": len(actual_future_any_pairs),

        "num_unique_candidate_pairs_actual_future_on_time": len(actual_future_on_time_pairs),

        "num_unique_candidate_pairs_true_path_any": len(true_path_any_pairs),

        "num_unique_candidate_pairs_true_path_on_time": len(true_path_on_time_pairs),

        "edge_candidate_precision_actual_future_any": _precision(len(actual_future_any_pairs)),

        "edge_candidate_precision_actual_future_on_time": _precision(len(actual_future_on_time_pairs)),

        "edge_candidate_precision_path_any": _precision(len(true_path_any_pairs)),

        "edge_candidate_precision_path_on_time": _precision(len(true_path_on_time_pairs)),

        "edge_first_ordered_ntmse": _edge_metric_mean("edge_first_ordered_ntse"),

        "edge_first_ordered_mae": _edge_metric_mean("edge_first_ordered_abs_time_error"),

        "edge_first_ordered_mean_lead_time": _edge_metric_mean("edge_first_ordered_lead_time"),

        "edge_first_ordered_mean_normalized_lead_time": _edge_metric_mean("edge_first_ordered_normalized_lead_time"),

        "edge_first_on_time_ntmse": _edge_metric_mean("edge_first_on_time_ntse"),

        "edge_first_on_time_mae": _edge_metric_mean("edge_first_on_time_abs_time_error"),

        "edge_first_on_time_mean_lead_time": _edge_metric_mean("edge_first_on_time_lead_time"),

        "edge_first_on_time_mean_normalized_lead_time": _edge_metric_mean("edge_first_on_time_normalized_lead_time"),

        **final_gt_path_info,

        **stress_summary,

    })

    return edge_rows, q, stress_rows










def audit_dataset(csv_name: str):

    csv_path = _resolve_existing_file(csv_name)

    file_stem = Path(csv_name).stem



    if ONLY_DATASET_STEMS is not None and file_stem not in ONLY_DATASET_STEMS:

        return None, None, None, None



    print(f"\n=== Standalone candidate-edge audit: {file_stem} ===")

    df = _normalize_df(csv_path)

    split_info = _compute_splits_like_main(df)

    pair_times = _pair_to_times(split_info["full_edges"])



    tests_df = _load_tests(file_stem)

    tests_df = tests_df[tests_df["time"].astype(int) > int(split_info["pivot_time"])].copy()

    if MAX_TEST_ROWS_PER_DATASET is not None:

        tests_df = tests_df.head(int(MAX_TEST_ROWS_PER_DATASET)).copy()



    print(

        f"pivot={split_info['pivot_time']} | future_times={len(split_info['future_times'])} | "

        f"queries={len(tests_df)} | DISABLE_ROOT_DIRECT_EDGES={DISABLE_ROOT_DIRECT_EDGES}"

    )



    active_fallback_target_pool = _build_global_activity_pool_from_prefix(

        edge_list=split_info["pivot_edges"],

        pool_size=DEST_CONNECTOR_ACTIVE_FALLBACK_SIZE,

    )

    workload_landmark_pool = _build_prefix_supported_landmark_pool(

        tests_df=tests_df,

        edge_list=split_info["pivot_edges"],

        pool_size=WORKLOAD_LANDMARK_POOL_SIZE,

        connector_pool_size=DEST_CONNECTOR_POOL_SIZE,

        active_fallback_pool=active_fallback_target_pool,

        active_fallback_size=DEST_CONNECTOR_ACTIVE_FALLBACK_SIZE,

    )

    landmark_connector_target_pool = _union_destination_connector_pools_from_prefix(

        edge_list=split_info["pivot_edges"],

        destinations=workload_landmark_pool,

        connector_pool_size=LANDMARK_CONNECTOR_POOL_SIZE,

        active_fallback_pool=active_fallback_target_pool,

        active_fallback_size=0,

    )



    print(

        f"prefix-supported landmarks={sorted(workload_landmark_pool)} | "

        f"landmark_connector_targets={len(landmark_connector_target_pool)}"

    )



    base_state_by_source = {}

    for src in sorted(set(int(x) for x in tests_df["source"].astype(int).tolist())):

        pivot_edges_for_src = list(split_info["pivot_edges"])

        pivot_edges_for_src.sort(key=lambda e: (e[2], e[0] != int(src)))



        (

            sp_pivot_base,

            L_pivot_base,

            _sp_pivot2_base,

            actual_pivot_base,

            dest_paths_pivot_base,

        ) = RES.computeActualShortestPathAndDistance(

            pivot_edges_for_src,

            int(src),

            split_info["vertex_list"],

            split_info["pivot_interval"],

        )

        pivot_baseline_sp_base = dict(sp_pivot_base)

        base_state_by_source[int(src)] = (

            sp_pivot_base,

            L_pivot_base,

            actual_pivot_base,

            dest_paths_pivot_base,

            pivot_baseline_sp_base,

        )



    all_edge_rows = []

    all_query_rows = []

    all_stress_rows = []

    for row in tests_df.itertuples(index=False):

        edge_rows, q, stress_rows = audit_one_query(

            row=row,

            dataset_stem=file_stem,

            split_info=split_info,

            pair_times=pair_times,

            base_state_by_source=base_state_by_source,

            active_fallback_target_pool=active_fallback_target_pool,

            workload_landmark_pool=workload_landmark_pool,

            landmark_connector_target_pool=landmark_connector_target_pool,

        )

        all_edge_rows.extend(edge_rows)

        all_query_rows.append(q)

        all_stress_rows.extend(stress_rows)



    edge_df = pd.DataFrame(all_edge_rows)

    query_df = pd.DataFrame(all_query_rows)

    stress_df = pd.DataFrame(all_stress_rows)



    edge_out = OUTPUT_DIR / f"{file_stem}_edge_level.tsv"

    query_out = OUTPUT_DIR / f"{file_stem}_query_level.tsv"

    stress_out = OUTPUT_DIR / f"{file_stem}_stress_step_level.tsv"

    edge_df.to_csv(edge_out, sep="\t", index=False)

    query_df.to_csv(query_out, sep="\t", index=False)

    stress_df.to_csv(stress_out, sep="\t", index=False)



    summary = {

        "dataset": file_stem,

        "landmark_pool_size": int(WORKLOAD_LANDMARK_POOL_SIZE),

        "num_queries": len(query_df),

        "num_future_only_edges": int(edge_df.shape[0]) if len(edge_df) else 0,

        "mean_edge_recall_any_by_query": query_df["edge_recall_any_by_query"].mean(skipna=True) if len(query_df) else None,

        "mean_edge_recall_on_time_any": query_df["edge_recall_on_time_any"].mean(skipna=True) if len(query_df) else None,

        "mean_edge_recall_ordered_oracle": query_df["edge_recall_ordered_oracle"].mean(skipna=True) if len(query_df) else None,

        "mean_ordered_prefix_recovery_rate": query_df["ordered_prefix_recovery_rate"].mean(skipna=True) if len(query_df) else None,

        "path_recoverable_ordered_oracle_rate": query_df["path_recoverable_ordered_oracle"].mean(skipna=True) if len(query_df) else None,

        "gt_path_present_final_nodes_only_rate": query_df["gt_path_present_final_nodes_only"].mean(skipna=True) if len(query_df) and "gt_path_present_final_nodes_only" in query_df else None,

        "gt_path_present_final_exact_future_edges_rate": query_df["gt_path_present_final_exact_future_edges"].mean(skipna=True) if len(query_df) and "gt_path_present_final_exact_future_edges" in query_df else None,

        "edge_level_found_any_rate": edge_df["candidate_found_any_by_query"].mean(skipna=True) if len(edge_df) else None,

        "edge_level_found_on_time_rate": edge_df["candidate_found_before_or_at_actual_time"].mean(skipna=True) if len(edge_df) else None,

        "edge_level_found_ordered_rate": edge_df["candidate_found_ordered_oracle"].mean(skipna=True) if len(edge_df) else None,

        "mean_num_unique_candidate_pairs_by_query": _safe_mean(query_df, "num_unique_candidate_pairs_by_query"),

        "mean_num_unique_candidate_pairs_actual_future_on_time": _safe_mean(query_df, "num_unique_candidate_pairs_actual_future_on_time"),

        "mean_num_unique_candidate_pairs_true_path_on_time": _safe_mean(query_df, "num_unique_candidate_pairs_true_path_on_time"),

        "mean_edge_candidate_precision_actual_future_any": _safe_mean(query_df, "edge_candidate_precision_actual_future_any"),

        "mean_edge_candidate_precision_actual_future_on_time": _safe_mean(query_df, "edge_candidate_precision_actual_future_on_time"),

        "mean_edge_candidate_precision_path_any": _safe_mean(query_df, "edge_candidate_precision_path_any"),

        "mean_edge_candidate_precision_path_on_time": _safe_mean(query_df, "edge_candidate_precision_path_on_time"),

        "mean_edge_first_ordered_ntmse": _safe_mean(query_df, "edge_first_ordered_ntmse"),

        "mean_edge_first_ordered_mae": _safe_mean(query_df, "edge_first_ordered_mae"),

        "mean_edge_first_ordered_lead_time": _safe_mean(query_df, "edge_first_ordered_mean_lead_time"),

        "mean_edge_first_ordered_normalized_lead_time": _safe_mean(query_df, "edge_first_ordered_mean_normalized_lead_time"),

        "mean_edge_first_on_time_ntmse": _safe_mean(query_df, "edge_first_on_time_ntmse"),

        "mean_edge_first_on_time_mae": _safe_mean(query_df, "edge_first_on_time_mae"),

        "mean_edge_first_on_time_lead_time": _safe_mean(query_df, "edge_first_on_time_mean_lead_time"),

        "mean_edge_first_on_time_normalized_lead_time": _safe_mean(query_df, "edge_first_on_time_mean_normalized_lead_time"),

        "disable_root_direct_edges": int(DISABLE_ROOT_DIRECT_EDGES),

        "use_destination_connector_targets": int(USE_DESTINATION_CONNECTOR_TARGETS),

        "include_path_targets_in_connector_mode": int(INCLUDE_PATH_TARGETS_IN_CONNECTOR_MODE),

        "include_reached_nodes_as_completion_sources": int(INCLUDE_REACHED_NODES_AS_COMPLETION_SOURCES),




        "stress_enabled": int(RUN_ALL_CANDIDATE_STRESS_PASS),

        "mean_stress_total_s": _safe_mean(query_df, "stress_total_s"),

        "mean_stress_candidate_generation_s": _safe_mean(query_df, "stress_candidate_generation_s"),

        "mean_stress_predictsp_s": _safe_mean(query_df, "stress_predictsp_s"),

        "sum_stress_total_s": _safe_sum(query_df, "stress_total_s"),

        "sum_stress_candidate_generation_s": _safe_sum(query_df, "stress_candidate_generation_s"),

        "sum_stress_predictsp_s": _safe_sum(query_df, "stress_predictsp_s"),

        "mean_stress_num_rollout_timestamps": _safe_mean(query_df, "stress_num_rollout_timestamps"),

        "mean_stress_num_predictsp_calls": _safe_mean(query_df, "stress_num_predictsp_calls"),

        "mean_stress_sum_unique_candidate_pairs": _safe_mean(query_df, "stress_sum_unique_candidate_pairs"),

        "mean_stress_mean_unique_candidate_pairs": _safe_mean(query_df, "stress_mean_unique_candidate_pairs"),

        "max_stress_max_unique_candidate_pairs": _safe_max(query_df, "stress_max_unique_candidate_pairs"),

        "mean_stress_sum_candidate_occurrences": _safe_mean(query_df, "stress_sum_candidate_occurrences"),

        "mean_stress_sum_pred_stream_edges": _safe_mean(query_df, "stress_sum_pred_stream_edges"),

        "max_stress_max_pred_stream_edges": _safe_max(query_df, "stress_max_pred_stream_edges"),

        "stress_candidate_cap_hit_rate": _safe_mean(query_df, "stress_candidate_cap_hit"),



        "workload_landmarks": ",".join(map(str, sorted(workload_landmark_pool))),

        "landmark_connector_target_pool_size": len(landmark_connector_target_pool),

    }

    summary_df = pd.DataFrame([summary])

    summary_out = OUTPUT_DIR / f"{file_stem}_summary.tsv"

    summary_df.to_csv(summary_out, sep="\t", index=False)



    print(f"saved: {query_out}")

    print(f"saved: {edge_out}")

    print(f"saved: {stress_out}")

    print(f"saved: {summary_out}")

    print(summary_df.to_string(index=False))



    if len(query_df):

        failures = query_df[query_df["path_recoverable_ordered_oracle"] == 0]

        if len(failures):

            print("\nFirst missing ordered future edges:")

            print(failures[[

                "source", "destination", "query_time", "future_Path",

                "future_only_edges", "first_missing_ordered_edge_index",

                "edge_recall_any_by_query", "edge_recall_ordered_oracle",

            ]].head(15).to_string(index=False))



    return edge_df, query_df, stress_df, summary_df





def _run_one_landmark_setting(landmark_pool_size: int):

    global WORKLOAD_LANDMARK_POOL_SIZE, OUTPUT_DIR



    WORKLOAD_LANDMARK_POOL_SIZE = int(landmark_pool_size)

    OUTPUT_DIR = OUTPUT_BASE_DIR / f"landmarks_{WORKLOAD_LANDMARK_POOL_SIZE}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



    print("\n" + "=" * 80)

    print(f"LANDMARK SETTING: WORKLOAD_LANDMARK_POOL_SIZE={WORKLOAD_LANDMARK_POOL_SIZE}")

    print(f"OUTPUT_DIR={OUTPUT_DIR}")

    print("=" * 80)



    all_edges = []

    all_queries = []

    all_stress = []

    all_summaries = []

    setting_start_s = time_lib.perf_counter()



    for csv_name, _pivot in DATASETS:

        edge_df, query_df, stress_df, summary_df = audit_dataset(csv_name)

        if edge_df is not None and len(edge_df):

            edge_df = edge_df.copy()

            edge_df["landmark_pool_size"] = int(WORKLOAD_LANDMARK_POOL_SIZE)

            all_edges.append(edge_df)

        if query_df is not None and len(query_df):

            query_df = query_df.copy()

            query_df["landmark_pool_size"] = int(WORKLOAD_LANDMARK_POOL_SIZE)

            all_queries.append(query_df)

        if stress_df is not None and len(stress_df):

            stress_df = stress_df.copy()

            stress_df["landmark_pool_size"] = int(WORKLOAD_LANDMARK_POOL_SIZE)

            all_stress.append(stress_df)

        if summary_df is not None and len(summary_df):

            summary_df = summary_df.copy()

            summary_df["landmark_pool_size"] = int(WORKLOAD_LANDMARK_POOL_SIZE)

            all_summaries.append(summary_df)



    if all_edges:

        pd.concat(all_edges, ignore_index=True).to_csv(

            OUTPUT_DIR / "ALL_DATASETS_edge_level.tsv", sep="\t", index=False

        )

    if all_queries:

        pd.concat(all_queries, ignore_index=True).to_csv(

            OUTPUT_DIR / "ALL_DATASETS_query_level.tsv", sep="\t", index=False

        )

    if all_stress:

        pd.concat(all_stress, ignore_index=True).to_csv(

            OUTPUT_DIR / "ALL_DATASETS_stress_step_level.tsv", sep="\t", index=False

        )

    if not all_summaries:

        return None



    summary_all = pd.concat(all_summaries, ignore_index=True)

    summary_all["landmark_setting_wallclock_s"] = time_lib.perf_counter() - setting_start_s

    summary_all.to_csv(OUTPUT_DIR / "ALL_DATASETS_summary.tsv", sep="\t", index=False)

    print("\n=== ALL SUMMARY ===")

    print(summary_all.to_string(index=False))

    print(f"saved directory: {OUTPUT_DIR}")

    return summary_all





def main():

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    all_setting_summaries = []



    for landmark_pool_size in LANDMARK_POOL_SIZES_TO_RUN:

        summary_all = _run_one_landmark_setting(int(landmark_pool_size))

        if summary_all is not None and len(summary_all):

            all_setting_summaries.append(summary_all)



    if not all_setting_summaries:

        print("No summaries produced.")

        return



    combined = pd.concat(all_setting_summaries, ignore_index=True)

    combined_out = OUTPUT_BASE_DIR / "ALL_LANDMARK_SETTINGS_summary.tsv"

    combined.to_csv(combined_out, sep="\t", index=False)




    numeric_cols = [

        c for c in combined.columns

        if c not in {"dataset", "workload_landmarks", "landmark_pool_size"}

        and pd.api.types.is_numeric_dtype(combined[c])

    ]

    mean_by_landmark = (

        combined

        .groupby("landmark_pool_size", as_index=False)[numeric_cols]

        .mean(numeric_only=True)

    )

    mean_out = OUTPUT_BASE_DIR / "LANDMARK_SIZE_COMPARISON_mean_over_datasets.tsv"

    mean_by_landmark.to_csv(mean_out, sep="\t", index=False)



    compact_cols = [

        "landmark_pool_size",




        "mean_edge_recall_ordered_oracle",

        "path_recoverable_ordered_oracle_rate",

        "gt_path_present_final_exact_future_edges_rate",

        "edge_level_found_ordered_rate",




        "mean_stress_total_s",

        "mean_stress_candidate_generation_s",

        "mean_stress_predictsp_s",

        "mean_stress_sum_unique_candidate_pairs",

        "mean_stress_sum_pred_stream_edges",

        "max_stress_max_unique_candidate_pairs",

        "stress_candidate_cap_hit_rate",

    ]

    compact_cols = [c for c in compact_cols if c in mean_by_landmark.columns]

    compact = mean_by_landmark[compact_cols].copy()

    compact_out = OUTPUT_BASE_DIR / "LANDMARK_SIZE_COMPARISON_compact.tsv"

    compact.to_csv(compact_out, sep="\t", index=False)



    print("\n=== LANDMARK SIZE COMPARISON: MEAN OVER DATASETS ===")

    print(mean_by_landmark.to_string(index=False))

    print(f"saved: {combined_out}")

    print(f"saved: {mean_out}")

    print(f"saved: {compact_out}")



if __name__ == "__main__":

    main()

