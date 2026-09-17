from __future__ import annotations

import heapq
import math
import os
import sys
from bisect import bisect_right
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import main as runner
import pipeline_methods

DATASETS = ["enron", "email_eu", "collegemsg", "bitcoin", "dblp_edges"]
ORACLES = [
    "n2vlp_static",
    "tgn_max_time",
    "tgn_per_time",
    "jodie_frozen",
    "jodie_update",
]
ORACLE_LABELS = {
    "n2vlp_static": "N2VLP-Static",
    "tgn_max_time": "TGN-MaxTime",
    "tgn_per_time": "TGN-PerTime",
    "jodie_frozen": "JODIE-Frozen",
    "jodie_update": "JODIE-Update",
}
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
OUTPUT_ROOT = Path("Results/RQ6_path_processing_baselines")
FULL_PIPELINE_OUTPUT_ROOT = OUTPUT_ROOT / "full_pipeline"
METHOD_ORDER = ["DisTB", "EB", "Ours"]
START_TIME = -(10**30)
Candidate = Dict[str, Any]
TemporalEdge = Tuple[int, int, int, float]


def _make_candidate(
    nodes: Sequence[int], edges: Sequence[TemporalEdge], score: float, p_exist: float
) -> Candidate:
    nodes_tuple = tuple((int(x) for x in nodes))
    norm_edges = tuple(((int(u), int(v), int(t), float(p)) for u, v, t, p in edges))
    return {
        "d": max(0, len(nodes_tuple) - 1),
        "T": int(norm_edges[-1][2]) if norm_edges else START_TIME,
        "nodes": nodes_tuple,
        "edges": norm_edges,
        "score": float(score),
        "p_exist": float(p_exist),
    }


def _normalize_submitted_topk(
    topk_tuples: Sequence[Tuple[Any, ...]],
) -> List[Candidate]:
    out: List[Candidate] = []
    for item in topk_tuples:
        if len(item) < 6:
            raise ValueError(
                f"Expected submitted tuple (d, T, nodes, edges, p_sp, p_exist), got: {item}"
            )
        d, T, nodes, edges, p_sp, p_exist = item[:6]
        nodes_tuple = tuple((int(x) for x in list(nodes)))
        if not nodes_tuple:
            continue
        norm_edges: List[TemporalEdge] = []
        for edge in list(edges):
            if len(edge) >= 4:
                u, v, t, p = edge[:4]
            elif len(edge) == 3:
                u, v, t = edge
                p = 1.0
            else:
                raise ValueError(f"Unexpected temporal edge: {edge}")
            norm_edges.append((int(u), int(v), int(t), float(p)))
        cand = _make_candidate(
            nodes=nodes_tuple,
            edges=norm_edges,
            score=float(p_sp),
            p_exist=float(p_exist),
        )
        cand["d"] = int(d) if math.isfinite(float(d)) else math.inf
        cand["T"] = int(T) if math.isfinite(float(T)) else math.inf
        out.append(cand)
    return out


class ObservedTemporalIndex:

    def __init__(self, observed_edges: Sequence[TemporalEdge], pivot_time: int) -> None:
        self.pivot_time = int(pivot_time)
        observed_by_u: Dict[int, Dict[Tuple[int, int], TemporalEdge]] = defaultdict(
            dict
        )
        graph_nodes = set()
        for u, v, t, p in observed_edges:
            u = int(u)
            v = int(v)
            t = int(t)
            if t > self.pivot_time:
                continue
            key = (v, t)
            item = (u, v, t, float(p))
            previous = observed_by_u[u].get(key)
            if previous is None or float(item[3]) > float(previous[3]):
                observed_by_u[u][key] = item
            graph_nodes.add(u)
            graph_nodes.add(v)
        self.out: Dict[int, Tuple[TemporalEdge, ...]] = {}
        self.times: Dict[int, Tuple[int, ...]] = {}
        reverse_neighbors: Dict[int, set[int]] = defaultdict(set)
        flat_edges: List[TemporalEdge] = []
        for u, edge_map in observed_by_u.items():
            edges = tuple(sorted(edge_map.values(), key=lambda e: int(e[2])))
            if not edges:
                continue
            self.out[int(u)] = edges
            self.times[int(u)] = tuple((int(e[2]) for e in edges))
            flat_edges.extend(edges)
            for _a, v, _t, _p in edges:
                reverse_neighbors[int(v)].add(int(u))
        self.edges: Tuple[TemporalEdge, ...] = tuple(
            sorted(flat_edges, key=lambda e: int(e[2]))
        )
        self.nodes = frozenset((int(x) for x in graph_nodes))
        self.reverse_neighbors: Dict[int, Tuple[int, ...]] = {
            int(v): tuple((int(u) for u in preds))
            for v, preds in reverse_neighbors.items()
        }
        self.node_bits: Dict[int, int] = {
            int(node): 1 << idx for idx, node in enumerate(self.nodes)
        }
        self.next_bit_index = int(len(self.node_bits))


class TemporalAdjacency:

    def __init__(
        self,
        observed_edges: Sequence[TemporalEdge],
        accepted_edges: Sequence[Dict[str, Any]],
        pivot_time: int,
        query_time: int,
        observed_index: ObservedTemporalIndex | None = None,
    ) -> None:
        self.pivot_time = int(pivot_time)
        self.query_time = int(query_time)
        if observed_index is None:
            observed_index = ObservedTemporalIndex(
                observed_edges=observed_edges, pivot_time=self.pivot_time
            )
        elif int(observed_index.pivot_time) != self.pivot_time:
            raise ValueError(
                f"Observed index pivot mismatch: index={observed_index.pivot_time}, query={self.pivot_time}"
            )
        self.observed_index = observed_index
        future_by_u: Dict[int, Dict[Tuple[int, int], TemporalEdge]] = defaultdict(dict)
        future_reverse_neighbors: Dict[int, set[int]] = defaultdict(set)
        for edge in accepted_edges:
            u = int(edge["u"])
            v = int(edge["v"])
            t = int(edge["time"])
            p = float(edge.get("score", 0.0))
            if not self.pivot_time < t <= self.query_time:
                continue
            key = (v, t)
            item = (u, v, t, p)
            previous = future_by_u[u].get(key)
            if previous is None or p > float(previous[3]):
                future_by_u[u][key] = item
        self.future_out: Dict[int, Tuple[TemporalEdge, ...]] = {}
        self.future_times: Dict[int, Tuple[int, ...]] = {}
        flat_future_edges: List[TemporalEdge] = []
        for u, edge_map in future_by_u.items():
            edges = tuple(sorted(edge_map.values(), key=lambda e: int(e[2])))
            if not edges:
                continue
            self.future_out[int(u)] = edges
            self.future_times[int(u)] = tuple((int(e[2]) for e in edges))
            flat_future_edges.extend(edges)
            for _a, v, _t, _p in edges:
                future_reverse_neighbors[int(v)].add(int(u))
        self.future_edges: Tuple[TemporalEdge, ...] = tuple(
            sorted(flat_future_edges, key=lambda e: int(e[2]))
        )
        self.future_reverse_neighbors: Dict[int, Tuple[int, ...]] = {
            int(v): tuple((int(u) for u in preds))
            for v, preds in future_reverse_neighbors.items()
        }
        self._combined_out: Dict[int, Tuple[TemporalEdge, ...]] = {}
        self._combined_times: Dict[int, Tuple[int, ...]] = {}
        self._future_bits: Dict[int, int] = {}
        self._next_bit_index = int(self.observed_index.next_bit_index)

    def _combined_for(self, u: int) -> Tuple[Tuple[TemporalEdge, ...], Tuple[int, ...]]:
        u = int(u)
        cached = self._combined_out.get(u)
        if cached is not None:
            return (cached, self._combined_times[u])
        observed = self.observed_index.out.get(u, ())
        observed_times = self.observed_index.times.get(u, ())
        future = self.future_out.get(u, ())
        future_times = self.future_times.get(u, ())
        if not observed:
            combined = future
            times = future_times
        elif not future:
            combined = observed
            times = observed_times
        else:
            combined = observed + future
            times = observed_times + future_times
        self._combined_out[u] = combined
        self._combined_times[u] = times
        return (combined, times)

    def iter_after(self, u: int, last_time: int) -> Iterable[TemporalEdge]:
        edges, times = self._combined_for(int(u))
        start = bisect_right(times, int(last_time))
        for i in range(start, len(edges)):
            yield edges[i]

    def iter_after_reverse(self, u: int, last_time: int) -> Iterable[TemporalEdge]:
        edges, times = self._combined_for(int(u))
        start = bisect_right(times, int(last_time))
        for i in range(len(edges) - 1, start - 1, -1):
            yield edges[i]

    def bit_for(self, node: int) -> int:
        node = int(node)
        observed_bit = self.observed_index.node_bits.get(node)
        if observed_bit is not None:
            return int(observed_bit)
        future_bit = self._future_bits.get(node)
        if future_bit is None:
            future_bit = 1 << int(self._next_bit_index)
            self._next_bit_index += 1
            self._future_bits[node] = int(future_bit)
        return int(future_bit)

    def static_hop_distances_to(
        self, destination: int, max_hops: int
    ) -> Dict[int, int]:
        destination = int(destination)
        max_hops = int(max_hops)
        distances: Dict[int, int] = {destination: 0}
        queue = deque([destination])
        while queue:
            v = int(queue.popleft())
            current_distance = int(distances[v])
            if current_distance >= max_hops:
                continue
            for predecessors in (
                self.observed_index.reverse_neighbors.get(v, ()),
                self.future_reverse_neighbors.get(v, ()),
            ):
                for u in predecessors:
                    u = int(u)
                    if u in distances:
                        continue
                    distances[u] = current_distance + 1
                    queue.append(u)
        return distances


def _relevance_sets_fast(
    graph: TemporalAdjacency, destination: int
) -> Tuple[set[int], set[int], set[int]]:
    destination = int(destination)
    future_reachable: set[int] = {destination}
    queue = deque([destination])
    while queue:
        v = int(queue.popleft())
        for u in graph.future_reverse_neighbors.get(v, ()):
            u = int(u)
            if u not in future_reachable:
                future_reachable.add(u)
                queue.append(u)
    gateways: set[int] = set()
    for u, v, _t, _p in graph.future_edges:
        if int(v) in future_reachable:
            gateways.add(int(u))
    observed_relevant: set[int] = set(gateways)
    queue = deque(observed_relevant)
    while queue:
        v = int(queue.popleft())
        for u in graph.observed_index.reverse_neighbors.get(v, ()):
            u = int(u)
            if u not in observed_relevant:
                observed_relevant.add(u)
                queue.append(u)
    return (future_reachable, gateways, observed_relevant)


class BaselineSearchContext:

    def __init__(
        self,
        graph: TemporalAdjacency,
        source: int,
        destination: int,
        pivot_distance: int,
    ) -> None:
        self.graph = graph
        self.source = int(source)
        self.destination = int(destination)
        self.pivot_distance = int(pivot_distance)
        self.static_distances = graph.static_hop_distances_to(
            destination=self.destination, max_hops=self.pivot_distance
        )
        self.future_reachable, self.gateways, self.observed_relevant = (
            _relevance_sets_fast(graph, self.destination)
        )
        source_static = self.static_distances.get(self.source)
        self.feasible = bool(
            self.pivot_distance > 0
            and graph.future_edges
            and (source_static is not None)
            and (int(source_static) <= self.pivot_distance)
            and (self.source in self.observed_relevant or self.source in self.gateways)
        )
        self._completion_bounds_cache: Dict[
            Tuple[int, int, int, bool], Tuple[float, float]
        ] = {}
        if self.feasible:
            self.relaxed_completion_bounds(
                u=self.source,
                last_time=START_TIME,
                remaining_hops=self.pivot_distance,
                used_future=False,
            )

    def relaxed_completion_bounds(
        self, *, u: int, last_time: int, remaining_hops: int, used_future: bool
    ) -> Tuple[float, float]:
        u = int(u)
        last_time = int(last_time)
        remaining_hops = int(remaining_hops)
        used_future = bool(used_future)
        key = (u, last_time, remaining_hops, used_future)
        cached = self._completion_bounds_cache.get(key)
        if cached is not None:
            return cached
        if u == self.destination:
            if used_future and last_time > self.graph.pivot_time:
                result = (float(last_time), 1.0)
            else:
                result = (math.inf, 0.0)
            self._completion_bounds_cache[key] = result
            return result
        if remaining_hops <= 0:
            result = (math.inf, 0.0)
            self._completion_bounds_cache[key] = result
            return result
        static_remaining = self.static_distances.get(u)
        if static_remaining is None or int(static_remaining) > remaining_hops:
            result = (math.inf, 0.0)
            self._completion_bounds_cache[key] = result
            return result
        if used_future:
            if u not in self.future_reachable:
                result = (math.inf, 0.0)
                self._completion_bounds_cache[key] = result
                return result
        elif u not in self.observed_relevant and u not in self.gateways:
            result = (math.inf, 0.0)
            self._completion_bounds_cache[key] = result
            return result
        best_time = math.inf
        best_probability = 0.0
        for a, v, t, p in self.graph.iter_after(u, last_time):
            if int(a) != u:
                raise AssertionError("Adjacency source mismatch")
            v = int(v)
            t = int(t)
            child_static = self.static_distances.get(v)
            if child_static is None or int(child_static) > remaining_hops - 1:
                continue
            edge_is_future = t > self.graph.pivot_time
            child_used_future = bool(used_future or edge_is_future)
            if child_used_future:
                if v != self.destination and v not in self.future_reachable:
                    continue
            elif v not in self.observed_relevant and v not in self.gateways:
                continue
            child_time, child_probability = self.relaxed_completion_bounds(
                u=v,
                last_time=t,
                remaining_hops=remaining_hops - 1,
                used_future=child_used_future,
            )
            if not math.isfinite(child_time):
                continue
            best_time = min(best_time, float(child_time))
            p = float(p)
            if not math.isfinite(p):
                continue
            if p < 0.0 or p > 1.0 + 1e-12:
                raise ValueError(f"Edge probability outside [0,1]: edge={(a, v, t, p)}")
            p = min(1.0, max(0.0, p))
            best_probability = max(best_probability, p * float(child_probability))
        result = (float(best_time), float(best_probability))
        self._completion_bounds_cache[key] = result
        return result


def _baseline_state_priority(
    method_name: str,
    *,
    context: BaselineSearchContext,
    u: int,
    last_time: int,
    hops: int,
    probability: float,
    used_future: bool,
) -> Tuple[Any, ...] | None:
    remaining_hops = int(context.pivot_distance) - int(hops)
    if remaining_hops < 0:
        return None
    earliest_time, max_suffix_probability = context.relaxed_completion_bounds(
        u=int(u),
        last_time=int(last_time),
        remaining_hops=remaining_hops,
        used_future=bool(used_future),
    )
    if not math.isfinite(earliest_time):
        return None
    if method_name == "DisTB":
        remaining = context.static_distances.get(int(u))
        if remaining is None:
            return None
        return (int(hops) + int(remaining), int(earliest_time))
    if method_name == "EB":
        return (-(float(probability) * float(max_suffix_probability)),)
    raise ValueError(f"Unknown baseline method: {method_name}")


def _candidate_from_state(
    *, states: Sequence[Tuple[Any, ...]], state_id: int, source: int, method_name: str
) -> Candidate:
    edge_rev: List[TemporalEdge] = []
    current_id = int(state_id)
    while current_id >= 0:
        (
            _u,
            _last_time,
            parent_id,
            incoming_edge,
            _probability,
            _used_future,
            _visited_mask,
            _hops,
        ) = states[current_id]
        if incoming_edge is not None:
            edge_rev.append(incoming_edge)
        current_id = int(parent_id)
    edges = tuple(reversed(edge_rev))
    nodes = [int(source)] + [int(edge[1]) for edge in edges]
    probability = float(states[int(state_id)][4])
    return _make_candidate(
        nodes=nodes,
        edges=edges,
        score=probability if method_name == "EB" else np.nan,
        p_exist=probability,
    )


def enumerate_baseline_topk(
    *,
    context: BaselineSearchContext,
    method_name: str,
    top_k: int,
) -> List[Candidate]:
    graph = context.graph
    source = int(context.source)
    destination = int(context.destination)
    pivot_distance = int(context.pivot_distance)
    top_k = int(top_k)
    if top_k <= 0 or not context.feasible:
        return []
    initial_priority = _baseline_state_priority(
        method_name,
        context=context,
        u=source,
        last_time=START_TIME,
        hops=0,
        probability=1.0,
        used_future=False,
    )
    if initial_priority is None:
        return []
    states: List[Tuple[Any, ...]] = [
        (source, START_TIME, -1, None, 1.0, False, int(graph.bit_for(source)), 0)
    ]
    heap: List[Tuple[Tuple[Any, ...], int, int, int]] = []
    serial = 0
    heapq.heappush(heap, (initial_priority, 0, serial, 0))
    results: List[Candidate] = []
    seen_node_paths: set[Tuple[int, ...]] = set()
    while heap and len(results) < top_k:
        _priority, _neg_hops, _serial, state_id = heapq.heappop(heap)
        (
            u,
            last_time,
            _parent_id,
            _incoming_edge,
            probability,
            used_future,
            visited_mask,
            current_hops,
        ) = states[int(state_id)]
        u = int(u)
        last_time = int(last_time)
        probability = float(probability)
        used_future = bool(used_future)
        visited_mask = int(visited_mask)
        current_hops = int(current_hops)
        if u == destination:
            if (
                used_future
                and last_time > graph.pivot_time
                and (current_hops <= pivot_distance)
            ):
                candidate = _candidate_from_state(
                    states=states,
                    state_id=int(state_id),
                    source=source,
                    method_name=method_name,
                )
                node_key = tuple(candidate["nodes"])
                if node_key not in seen_node_paths:
                    seen_node_paths.add(node_key)
                    results.append(candidate)
            continue
        if current_hops >= pivot_distance:
            continue
        if not used_future:
            if u not in context.observed_relevant and u not in context.gateways:
                continue
        elif u not in context.future_reachable:
            continue
        for a, v, t, p in graph.iter_after(u, last_time):
            if int(a) != u:
                raise AssertionError("Adjacency source mismatch")
            v = int(v)
            t = int(t)
            new_hops = current_hops + 1
            if v == destination:
                if new_hops > pivot_distance:
                    continue
            elif new_hops >= pivot_distance:
                continue
            static_remaining = context.static_distances.get(v)
            if (
                static_remaining is None
                or new_hops + int(static_remaining) > pivot_distance
            ):
                continue
            bit = int(graph.bit_for(v))
            if visited_mask & bit:
                continue
            edge_is_future = t > graph.pivot_time
            new_used_future = bool(used_future or edge_is_future)
            if not new_used_future:
                if v not in context.observed_relevant and v not in context.gateways:
                    continue
            elif v != destination and v not in context.future_reachable:
                continue
            p = float(p)
            if not math.isfinite(p):
                continue
            if p < 0.0 or p > 1.0 + 1e-12:
                raise ValueError(f"Edge probability outside [0,1]: edge={(a, v, t, p)}")
            p = min(1.0, max(0.0, p))
            new_probability = probability * p
            child_priority = _baseline_state_priority(
                method_name,
                context=context,
                u=v,
                last_time=t,
                hops=new_hops,
                probability=new_probability,
                used_future=new_used_future,
            )
            if child_priority is None:
                continue
            new_state = (
                v,
                t,
                int(state_id),
                (u, v, t, p),
                float(new_probability),
                new_used_future,
                int(visited_mask | bit),
                new_hops,
            )
            states.append(new_state)
            new_state_id = len(states) - 1
            serial += 1
            heapq.heappush(heap, (child_priority, -new_hops, serial, new_state_id))
    return results


def _sequence_edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    a = [int(x) for x in a]
    b = [int(x) for x in b]
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (0 if x == y else 1),
                )
            )
        previous = current
    return int(previous[-1])


def _path_edit_error(
    predicted_nodes: Sequence[int], true_nodes: Sequence[int]
) -> float:
    predicted_nodes = list(predicted_nodes)
    true_nodes = list(true_nodes)
    denominator = max(len(predicted_nodes), len(true_nodes), 1)
    return float(_sequence_edit_distance(predicted_nodes, true_nodes) / denominator)


def _normalized_distance_mse(
    predicted_nodes: Sequence[int], true_nodes: Sequence[int]
) -> float:
    predicted_len = max(0, len(predicted_nodes) - 1)
    true_len = max(0, len(true_nodes) - 1)
    denominator = max(predicted_len, true_len, 1) ** 2
    return float((predicted_len - true_len) ** 2 / denominator)


def _future_valid_complete_prefix(
    ranked: Sequence[Candidate], pivot_time: int, top_k: int
) -> List[Candidate]:
    prefix = list(ranked)[: int(top_k)]
    return [
        cand
        for cand in prefix
        if cand.get("T", math.inf) != math.inf and int(cand["T"]) > int(pivot_time)
    ]


def _metrics_for_ranked_candidates(
    ranked: Sequence[Candidate],
    true_nodes: Sequence[int],
    pivot_time: int,
    top_k: int,
) -> Dict[str, Any]:
    future_ranked = _future_valid_complete_prefix(
        ranked, pivot_time=pivot_time, top_k=top_k
    )
    true_nodes = [int(node) for node in true_nodes]
    true_key = tuple(true_nodes)
    hit = any((tuple(candidate["nodes"]) == true_key for candidate in future_ranked))
    pee_values = [
        _path_edit_error(candidate["nodes"], true_nodes) for candidate in future_ranked
    ]
    ndmse_values = [
        _normalized_distance_mse(candidate["nodes"], true_nodes)
        for candidate in future_ranked
    ]
    return {
        "future_coverage": float(bool(future_ranked)),
        "mean_pee_at_10": float(np.mean(pee_values)) if pee_values else np.nan,
        "mean_ndmse_at_10": float(np.mean(ndmse_values)) if ndmse_values else np.nan,
        "path_recall_at_10": float(hit),
    }


_ORIGINAL_PREPARE_DATASET = pipeline_methods.prepare_dataset
_ORIGINAL_FRESH_QUERY_STATE = pipeline_methods._fresh_query_state
_ORIGINAL_TOPK_ROWS = pipeline_methods._topk_prediction_rows_for_query
_ORIGINAL_ACCEPTED_ROWS = pipeline_methods._accepted_edge_rows_for_query
CURRENT_OBSERVED_INDEX: ObservedTemporalIndex | None = None
CURRENT_PIVOT_BASELINE_SP: Dict[int, Any] | None = None
CURRENT_ORACLE_LABEL = ""
CAPTURED_SUBMITTED_TOPK: Dict[Tuple[Any, ...], List[Tuple[Any, ...]]] = {}
CAPTURED_PIVOT_DISTANCE: Dict[Tuple[Any, ...], int] = {}
BASELINE_QUERY_ROWS: List[Dict[str, Any]] = []


def _capturing_prepare_dataset(*args, **kwargs):
    global CURRENT_OBSERVED_INDEX
    global CURRENT_PIVOT_BASELINE_SP
    data = _ORIGINAL_PREPARE_DATASET(*args, **kwargs)
    CURRENT_PIVOT_BASELINE_SP = None
    CAPTURED_SUBMITTED_TOPK.clear()
    CAPTURED_PIVOT_DISTANCE.clear()
    CURRENT_OBSERVED_INDEX = None
    if data is not None:
        CURRENT_OBSERVED_INDEX = ObservedTemporalIndex(
            observed_edges=data["pivot_edges"], pivot_time=int(data["pivot_time"])
        )
    return data


def _capturing_accepted_edge_rows(
    dataset: str,
    file_stem: str,
    iteration: int,
    method: str,
    row: Any,
    pivot_time: int,
    accepted_edges: Sequence[Dict[str, Any]],
):
    if CURRENT_OBSERVED_INDEX is None:
        raise RuntimeError("Dataset capture is unavailable.")
    key = _query_key(dataset, file_stem, iteration, method, row, pivot_time)
    if key not in CAPTURED_SUBMITTED_TOPK or key not in CAPTURED_PIVOT_DISTANCE:
        raise RuntimeError(f"Incomplete capture for query key={key}")
    pivot_distance = CAPTURED_PIVOT_DISTANCE.pop(key)
    submitted_topk = _normalize_submitted_topk(CAPTURED_SUBMITTED_TOPK.pop(key))
    query_time = int(row.time)
    true_nodes = pipeline_methods.split_path_str(row.future_Path)
    accepted_valid = [
        {
            "u": int(edge["u"]),
            "v": int(edge["v"]),
            "time": int(edge["time"]),
            "score": float(edge.get("score", 0.0)),
        }
        for edge in accepted_edges
        if int(pivot_time) < int(edge["time"]) <= query_time
    ]
    graph = TemporalAdjacency(
        observed_edges=(),
        accepted_edges=accepted_valid,
        pivot_time=int(pivot_time),
        query_time=query_time,
        observed_index=CURRENT_OBSERVED_INDEX,
    )
    context = BaselineSearchContext(
        graph=graph,
        source=int(row.source),
        destination=int(row.destination),
        pivot_distance=int(pivot_distance),
    )
    methods = []
    for method_name in ("DisTB", "EB"):
        ranked = enumerate_baseline_topk(
            context=context,
            method_name=method_name,
            top_k=TOP_K,
        )
        methods.append((method_name, ranked))
    methods.append(("Ours", submitted_topk))
    for method_name, ranked in methods:
        metrics = _metrics_for_ranked_candidates(
            ranked=ranked,
            true_nodes=true_nodes,
            pivot_time=int(pivot_time),
            top_k=TOP_K,
        )
        BASELINE_QUERY_ROWS.append(
            {
                "dataset": dataset,
                "oracle": CURRENT_ORACLE_LABEL,
                "iteration": int(iteration),
                "method": method_name,
                "source": int(row.source),
                "destination": int(row.destination),
                "query_time": query_time,
                **metrics,
            }
        )
    return _ORIGINAL_ACCEPTED_ROWS(
        dataset=dataset,
        file_stem=file_stem,
        iteration=iteration,
        method=method,
        row=row,
        pivot_time=pivot_time,
        accepted_edges=accepted_edges,
    )


def _query_key(
    dataset: str, file_stem: str, iteration: int, method: str, row: Any, pivot_time: int
) -> Tuple[Any, ...]:
    return (
        str(dataset),
        str(file_stem),
        int(iteration),
        str(method),
        int(row.source),
        int(row.destination),
        int(row.time),
        int(pivot_time),
    )


def _capturing_fresh_query_state(*args, **kwargs):
    global CURRENT_PIVOT_BASELINE_SP
    result = _ORIGINAL_FRESH_QUERY_STATE(*args, **kwargs)
    if not isinstance(result, tuple) or len(result) < 5:
        raise RuntimeError(
            "Unexpected _fresh_query_state return shape. Expected at least five values including pivot_baseline_sp."
        )
    CURRENT_PIVOT_BASELINE_SP = result[4]
    return result


def _pivot_distance_from_current_state(destination: int) -> int:
    if CURRENT_PIVOT_BASELINE_SP is None:
        raise RuntimeError("Pivot baseline state was not captured before top-k output.")
    destination = int(destination)
    pivot_entry = CURRENT_PIVOT_BASELINE_SP.get(destination, (math.inf, [], 0, 0.0))
    pivot_dist = pivot_entry[0]
    pivot_nodes = pivot_entry[1] if len(pivot_entry) > 1 else []
    if not math.isfinite(float(pivot_dist)) or not pivot_nodes:
        raise RuntimeError(
            f"No finite pivot path is available for the baseline distance bound: destination={destination}, pivot_entry={pivot_entry}"
        )
    pivot_distance = int(pivot_dist)
    node_distance = len(pivot_nodes) - 1
    if node_distance != pivot_distance:
        raise RuntimeError(
            f"Pivot distance/path mismatch in repository state: distance={pivot_distance}, len(path)-1={node_distance}, destination={destination}"
        )
    if pivot_distance <= 0:
        raise RuntimeError(f"Invalid non-positive pivot distance: {pivot_distance}")
    return pivot_distance


def _capturing_topk_prediction_rows(
    dataset: str,
    file_stem: str,
    iteration: int,
    method: str,
    row: Any,
    pivot_time: int,
    topk_tuples: Sequence[Tuple[Any, ...]],
    max_rank: int,
):
    key = _query_key(dataset, file_stem, iteration, method, row, pivot_time)
    CAPTURED_SUBMITTED_TOPK[key] = [tuple(item) for item in topk_tuples]
    CAPTURED_PIVOT_DISTANCE[key] = _pivot_distance_from_current_state(
        int(row.destination)
    )
    return _ORIGINAL_TOPK_ROWS(
        dataset=dataset,
        file_stem=file_stem,
        iteration=iteration,
        method=method,
        row=row,
        pivot_time=pivot_time,
        topk_tuples=topk_tuples,
        max_rank=max_rank,
    )


def install_capture_hooks() -> None:
    pipeline_methods.prepare_dataset = _capturing_prepare_dataset
    pipeline_methods._fresh_query_state = _capturing_fresh_query_state
    pipeline_methods._topk_prediction_rows_for_query = _capturing_topk_prediction_rows
    pipeline_methods._accepted_edge_rows_for_query = _capturing_accepted_edge_rows


def uninstall_capture_hooks() -> None:
    pipeline_methods.prepare_dataset = _ORIGINAL_PREPARE_DATASET
    pipeline_methods._fresh_query_state = _ORIGINAL_FRESH_QUERY_STATE
    pipeline_methods._topk_prediction_rows_for_query = _ORIGINAL_TOPK_ROWS
    pipeline_methods._accepted_edge_rows_for_query = _ORIGINAL_ACCEPTED_ROWS


def safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else np.nan


def write_outputs() -> None:
    if not BASELINE_QUERY_ROWS:
        return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    query_df = pd.DataFrame(BASELINE_QUERY_ROWS)
    query_df.to_csv(
        OUTPUT_ROOT / "path_processing_query_metrics.tsv", sep="\t", index=False
    )
    iteration_rows = []
    for keys, group in query_df.groupby(
        ["dataset", "oracle", "iteration", "method"], sort=False
    ):
        dataset, oracle_name, iteration, method_name = keys
        covered = pd.to_numeric(group["future_coverage"], errors="coerce") == 1.0
        iteration_rows.append(
            {
                "dataset": dataset,
                "oracle": oracle_name,
                "iteration": int(iteration),
                "method": method_name,
                "queries": len(group),
                "covered_queries": int(covered.sum()),
                "mean_pee_at_10": safe_mean(group.loc[covered, "mean_pee_at_10"]),
                "mean_ndmse_at_10": safe_mean(group.loc[covered, "mean_ndmse_at_10"]),
                "path_recall_at_10": safe_mean(group["path_recall_at_10"]),
            }
        )
    iteration_df = pd.DataFrame(iteration_rows)
    iteration_df.to_csv(
        OUTPUT_ROOT / "path_processing_iteration_metrics.tsv", sep="\t", index=False
    )
    summary_rows = []
    for keys, group in iteration_df.groupby(
        ["dataset", "oracle", "method"], sort=False
    ):
        dataset, oracle_name, method_name = keys
        summary_rows.append(
            {
                "dataset": dataset,
                "oracle": oracle_name,
                "method": method_name,
                "runs": int(group["iteration"].nunique()),
                "queries": int(round(group["queries"].mean())),
                "mean_pee_at_10": safe_mean(group["mean_pee_at_10"]),
                "mean_ndmse_at_10": safe_mean(group["mean_ndmse_at_10"]),
                "path_recall_at_10": safe_mean(group["path_recall_at_10"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df["method"] = pd.Categorical(
        summary_df["method"], categories=METHOD_ORDER, ordered=True
    )
    summary_df = summary_df.sort_values(["dataset", "oracle", "method"]).reset_index(
        drop=True
    )
    summary_df.to_csv(
        OUTPUT_ROOT / "path_processing_summary.tsv", sep="\t", index=False
    )
    print(summary_df.round(4).to_string(index=False))


def register_dataset(dataset: str) -> None:
    if dataset != "dblp_edges":
        return
    runner.DATASET_FILES.setdefault("dblp_edges", "dblp_edges.csv")
    runner.QUERY_TEST_FILES.setdefault("dblp_edges", "dblp_edges.tsv")


def run_one(dataset: str, oracle_name: str) -> None:
    global CURRENT_ORACLE_LABEL
    register_dataset(dataset)
    runner.DATASET_NAME = dataset
    runner.ORACLE_NAME = oracle_name
    runner.DATA_ROOT = DATA_ROOT
    runner.QUERY_TEST_ROOT = QUERY_TEST_ROOT
    runner.OUTPUT_ROOT = str(FULL_PIPELINE_OUTPUT_ROOT / oracle_name / dataset)
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
    CURRENT_ORACLE_LABEL = ORACLE_LABELS[oracle_name]
    runner.main()
    if CAPTURED_SUBMITTED_TOPK or CAPTURED_PIVOT_DISTANCE:
        raise RuntimeError(
            f"Unmatched captured queries remain for {dataset} and {oracle_name}."
        )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FULL_PIPELINE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    install_capture_hooks()
    try:
        for dataset in DATASETS:
            for oracle_name in ORACLES:
                run_one(dataset, oracle_name)
                write_outputs()
    finally:
        uninstall_capture_hooks()


if __name__ == "__main__":
    main()
