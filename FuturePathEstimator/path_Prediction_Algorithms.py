import Luby_KarpAlgorithm as LK

import math

import time as time_lib

import heapq





def _topk_unique_node_paths_by_psp(candidates, k):

    k = max(1, int(k))

    best_by_nodes = {}



    def rank_key(cand):



        return (

            -float(cand[4]),

            int(cand[0]) if math.isfinite(cand[0]) else math.inf,

            int(cand[1]) if math.isfinite(cand[1]) else math.inf,

            -float(cand[5]),

        )



    for cand in candidates:

        node_key = tuple(cand[2])

        old = best_by_nodes.get(node_key)



        if old is None or rank_key(cand) < rank_key(old):

            best_by_nodes[node_key] = cand



    if not best_by_nodes:

        return []



    return heapq.nsmallest(k, best_by_nodes.values(), key=rank_key)





def predictBiggestSPDistanceShortestPathAndDistance(

    sortedEdgeStream,

    vertexToLv_dict,

    shortestPath_dict,

    actualPath_dict,

    vertex,

    vertexList,

    timeInterval,

    destinationPathDistance_Dict,

    existenceBound,

    shortestProbaBound=0.0,

    lk_runs=20,

    pivot_time=None,

    pivot_baseline_sp=None,

    return_timing=False,

    top_k_result_paths=1,

    return_topk=False,

):



    _predictsp_total_start = time_lib.perf_counter()

    timing_info = {

        "predictsp_total_s": 0.0,

        "predictsp_stream_update_s": 0.0,

        "by_dst": {},

    }



    t_min, t_max = timeInterval

    top_k_result_paths = max(1, int(top_k_result_paths))





    def make_record(d, T, p, nodes, edges):

        return {

            "d": d,

            "T": T,

            "p": p,

            "nodes": list(nodes),

            "edges": list(edges),

        }



    def prune_skyline(Lv, max_extra_dominated=4):





        best = {}

        for rec in Lv:

            edge_sig = tuple((a, b, tt) for (a, b, tt, _) in rec["edges"])

            key = (len(rec["nodes"]) - 1, rec["T"], edge_sig)

            if key not in best or rec["p"] > best[key]["p"]:

                best[key] = rec



        items = list(best.values())



        skyline = []

        dominated = []



        for rec in items:

            d = len(rec["nodes"]) - 1

            T = rec["T"]

            p = rec["p"]



            is_dominated = any(

                ((len(other["nodes"]) - 1) <= d and other["T"] <= T and other["p"] >= p) and

                ((len(other["nodes"]) - 1) < d or other["T"] < T or other["p"] > p)

                for other in items if other is not rec

            )



            if is_dominated:

                dominated.append(rec)

            else:

                skyline.append(rec)



        skyline.sort(key=lambda r: (len(r["nodes"]) - 1, r["T"], -r["p"]))

        dominated.sort(key=lambda r: (len(r["nodes"]) - 1, r["T"], -r["p"]))



        return skyline + dominated[:max_extra_dominated]



    def get_best_record(Lv):

        if not Lv:

            return None

        return min(Lv, key=lambda r: (len(r["nodes"]) - 1, r["T"], -r["p"]))





    for v in vertexList:

        shortestPath_dict.setdefault(v, (math.inf, [], 0, 0.0))

        actualPath_dict.setdefault(v, ([], []))

        vertexToLv_dict.setdefault(v, [])



    if not vertexToLv_dict[vertex]:

        vertexToLv_dict[vertex] = [

            make_record(0, t_min, 1.0, [vertex], [])

        ]

        actualPath_dict[vertex] = ([vertex], [])





    _stream_update_start = time_lib.perf_counter()

    for (u, v, t, p_edge) in sortedEdgeStream:





        if u not in vertexList or v not in vertexList:

            continue

        if not (t_min <= t <= t_max):

            if t > t_max:

                break

            continue





        L_u = vertexToLv_dict.get(u, [])

        if not L_u:

            continue



        new_records_for_v = []

        for rec_u in L_u:

            if rec_u["T"] >= t:

                continue





            if v in rec_u["nodes"]:

                continue



            p_new = rec_u["p"] * p_edge

            if p_new < existenceBound:

                continue



            rec_v = make_record(

                d=len(rec_u["nodes"] + [v]) - 1,

                T=t,

                p=p_new,

                nodes=rec_u["nodes"] + [v],

                edges=rec_u["edges"] + [(u, v, t, p_edge)],

            )

            new_records_for_v.append(rec_v)



        if not new_records_for_v:

            continue





        L_v = vertexToLv_dict.get(v, [])

        L_v.extend(new_records_for_v)





        L_v = [rec for rec in L_v if rec["p"] >= existenceBound]

        L_v = prune_skyline(L_v)

        vertexToLv_dict[v] = L_v





        best_rec = get_best_record(L_v)

        if best_rec is not None:

            shortestPath_dict[v] = (

                len(best_rec["nodes"]) - 1,

                best_rec["nodes"],

                best_rec["T"],

                best_rec["p"],

            )





        candidates = []

        for rec in L_v:

            if len(rec["nodes"]) >= 2:

                candidates.append(

                    (len(rec["nodes"]) - 1, rec["T"], rec["nodes"], rec["edges"], rec["p"])

                )





        best_by_path = {}

        for cand in candidates:

            d, T, nodes, edges, p = cand

            edge_sig = tuple((a, b, tt) for (a, b, tt, _) in edges)

            key = (edge_sig, d, T)

            if key not in best_by_path or p > best_by_path[key][4]:

                best_by_path[key] = cand



        destinationPathDistance_Dict[v] = sorted(

            best_by_path.values(),

            key=lambda x: (len(x[2]) - 1, x[1], -x[4])

        )

    timing_info["predictsp_stream_update_s"] = time_lib.perf_counter() - _stream_update_start





    shortestPath_dict[vertex] = (0, [vertex], t_min, 1.0)





    biggestShortestPathProba_dict = {}

    topKShortestPathProba_dict = {}

    candidateCount_dict = {}



    for dst, value in destinationPathDistance_Dict.items():

        dst_timing = {

            "predictsp_create_candidate_paths_s": 0.0,

            "predictsp_lower_bound_s": 0.0,

            "predictsp_luby_karp_s": 0.0,

            "predictsp_choose_shortest_s": 0.0,

            "predictsp_candidate_paths_before_lb": 0,

            "predictsp_candidate_paths_after_lb": 0,

            "predictsp_topk_returned": 0,

        }

        timing_info["by_dst"][dst] = dst_timing



        if not value:

            candidateCount_dict[dst] = (0, 0)

            emptyPath = (math.inf, math.inf, [], [], 0.0, 0.0)

            biggestShortestPathProba_dict[dst] = emptyPath

            topKShortestPathProba_dict[dst] = []

            continue





        _create_candidate_paths_start = time_lib.perf_counter()

        candidatePathList = []

        seen_edge_seqs = set()

        possiblePathsList = []



        for item in value:

            if isinstance(item, dict):

                d = item["d"]

                T = item["T"]

                nodes = item["nodes"]

                edges = item["edges"]

                p_exist = item.get("prob", item.get("p", 1.0))

            else:

                d, T, nodes, edges, p_exist = item



            edge_tuple = tuple((a, b, tt) for (a, b, tt, _) in edges)

            if edge_tuple in seen_edge_seqs:

                continue



            seen_edge_seqs.add(edge_tuple)

            candidatePathList.append(edges)

            possiblePathsList.append((d, T, nodes, edges, p_exist))



        dst_timing["predictsp_create_candidate_paths_s"] = (

            time_lib.perf_counter() - _create_candidate_paths_start

        )



        if dst != vertex and not candidatePathList:

            candidateCount_dict[dst] = (0, 0)

            emptyPath = (math.inf, math.inf, [], [], 0.0, 0.0)

            biggestShortestPathProba_dict[dst] = emptyPath

            topKShortestPathProba_dict[dst] = []

            continue





        if dst == vertex:

            candidateCount_dict[dst] = (1, 1)

            dst_timing["predictsp_candidate_paths_before_lb"] = 1

            dst_timing["predictsp_candidate_paths_after_lb"] = 1

            chosenPath = (0, t_min, [vertex], [], 1.0, 1.0)

            chosenPaths = [chosenPath]

            dst_timing["predictsp_topk_returned"] = 1

        else:

            before_lb_count = len(candidatePathList)

            dst_timing["predictsp_candidate_paths_before_lb"] = before_lb_count



            if len(candidatePathList) > 1:

                _lower_bound_start = time_lib.perf_counter()

                lowerBoundList = []

                for i in range(len(candidatePathList)):

                    lb = LK.computeFirstLowerBound(candidatePathList, i)

                    lowerBoundList.append(lb)

                dst_timing["predictsp_lower_bound_s"] = time_lib.perf_counter() - _lower_bound_start



                keep_idx = [i for i, lb in enumerate(lowerBoundList) if lb > shortestProbaBound]

                if len(keep_idx) >= 1:

                    candidatePathList = [candidatePathList[i] for i in keep_idx]

                    possiblePathsList = [possiblePathsList[i] for i in keep_idx]

                else:





                    pivot_idx = [

                        i for i, item in enumerate(possiblePathsList)

                        if pivot_time is not None and int(item[1]) <= int(pivot_time)

                    ]



                    if pivot_idx:

                        best_pivot_idx = min(

                            pivot_idx,

                            key=lambda i: (

                                possiblePathsList[i][0],

                                possiblePathsList[i][1],

                                -possiblePathsList[i][4],

                            )

                        )



                        candidatePathList = [candidatePathList[best_pivot_idx]]

                        possiblePathsList = [possiblePathsList[best_pivot_idx]]

                    else:

                        candidatePathList = []

                        possiblePathsList = []



            after_lb_count = len(candidatePathList)

            dst_timing["predictsp_candidate_paths_after_lb"] = after_lb_count

            candidateCount_dict[dst] = (before_lb_count, after_lb_count)

            if not candidatePathList:

                emptyPath = (

                    math.inf,

                    math.inf,

                    [],

                    [],

                    0.0,

                    0.0,

                )

                biggestShortestPathProba_dict[dst] = emptyPath

                topKShortestPathProba_dict[dst] = []

                actualPath_dict[dst] = ([], [])

                continue



            shortestPathsProbabilityList = []

            _lk_start = time_lib.perf_counter()

            for i in range(len(candidatePathList)):

                est = LK.LubyKarpSampling(candidatePathList, lk_runs, i)

                d, T, nodes, edges, p_exist = possiblePathsList[i]

                cand_len = len(nodes) - 1

                shortestPathsProbabilityList.append((cand_len, T, nodes, edges, est, p_exist))

            dst_timing["predictsp_luby_karp_s"] = time_lib.perf_counter() - _lk_start



            _choose_start = time_lib.perf_counter()



            chosenPaths = _topk_unique_node_paths_by_psp(

                shortestPathsProbabilityList,

                top_k_result_paths,

            )

            chosenPath = chosenPaths[0]

            dst_timing["predictsp_topk_returned"] = len(chosenPaths)

            dst_timing["predictsp_choose_shortest_s"] = time_lib.perf_counter() - _choose_start



        biggestShortestPathProba_dict[dst] = chosenPath

        topKShortestPathProba_dict[dst] = chosenPaths



        actualPath_dict[dst] = (chosenPath[2], chosenPath[3])



    timing_info["predictsp_total_s"] = time_lib.perf_counter() - _predictsp_total_start



    if return_topk:

        result = (

            shortestPath_dict,

            vertexToLv_dict,

            actualPath_dict,

            destinationPathDistance_Dict,

            biggestShortestPathProba_dict,

            topKShortestPathProba_dict,

            candidateCount_dict,

        )

    else:

        result = (

            shortestPath_dict,

            vertexToLv_dict,

            actualPath_dict,

            destinationPathDistance_Dict,

            biggestShortestPathProba_dict,

            candidateCount_dict,

        )



    if return_timing:

        return result + (timing_info,)

    return result





def computeShortestPathProbabilitySampling(

    PathList,

    candidatePathList,

    shortestProbaBound,

    lk_runs,

    top_k_result_paths=1,

    return_topk=False,

):

    top_k_result_paths = max(1, int(top_k_result_paths))



    shortestPathsProbabilityList = []

    pathsToDeleteList = []



    for i, pathSet in enumerate(PathList):

        d, T, nodes, edges, p_exist = pathSet



        expectedShortestPathProbability = LK.LubyKarpSampling(candidatePathList, lk_runs, i)

        if expectedShortestPathProbability < shortestProbaBound:

            pathsToDeleteList.append(pathSet)

        shortestPathsProbabilityList.append((d, T, nodes, edges, expectedShortestPathProbability, p_exist))





    topKPaths = _topk_unique_node_paths_by_psp(

        shortestPathsProbabilityList,

        top_k_result_paths,

    )

    if return_topk:

        return topKPaths[0], topKPaths, PathList

    return topKPaths[0], PathList

