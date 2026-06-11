import random





def _candidate_meta(path):

    length = len(path)

    formation_time = path[-1][2] if path else float("inf")

    return length, formation_time





def _is_competitor_of(candidate_list, other_idx, cand_idx):

    if other_idx == cand_idx:

        return False



    other_len, other_T = _candidate_meta(candidate_list[other_idx])

    cand_len, cand_T = _candidate_meta(candidate_list[cand_idx])



    if other_len < cand_len:

        return True

    if other_len == cand_len and other_T < cand_T:

        return True

    if other_len == cand_len and other_T == cand_T and other_idx < cand_idx:

        return True



    return False





def MonteCarloEstimator(CandidatePathsList, LKRunNumber, indexofCandidateLength):

    Pd = CandidatePathsList[indexofCandidateLength]

    competitor_indices = [

        i for i in range(len(CandidatePathsList))

        if _is_competitor_of(CandidatePathsList, i, indexofCandidateLength)

    ]

    competitor_paths = [CandidatePathsList[i] for i in competitor_indices]





    edge_prob = {}

    def add_path(path):

        for (u, v, t, p) in path:

            key = (u, v, t)



            edge_prob[key] = max(edge_prob.get(key, 0.0), p)



    add_path(Pd)

    for P in competitor_paths:

        add_path(P)





    Pd_keys = {(u, v, t) for (u, v, t, _) in Pd}

    shorter_keys = [ {(u, v, t) for (u, v, t, _) in P} for P in competitor_paths ]

    edge_keys = list(edge_prob.keys())



    success = 0



    for _ in range(LKRunNumber):



        realized = {}

        for key in edge_keys:

            p = edge_prob[key]

            realized[key] = (random.random() < p)





        Pd_exists = all(realized[k] for k in Pd_keys)

        if not Pd_exists:

            continue





        shorter_exists = False

        for K in shorter_keys:



            if all(realized.get(k, False) for k in K):

                shorter_exists = True

                break



        if not shorter_exists:

            success += 1





    return success / float(LKRunNumber) if LKRunNumber > 0 else 0.0





def _edge_key(edge):



    return (edge[0], edge[1], edge[2])





def _path_exist_probability(path):

    prob = 1.0

    for _, _, _, p in path:

        prob *= p

    return prob





def _private_edges(Pi, Pd):

    Pd_keys = {_edge_key(e) for e in Pd}

    return [e for e in Pi if _edge_key(e) not in Pd_keys]





def _event_probability(edge_list):

    prob = 1.0

    for _, _, _, p in edge_list:

        prob *= p

    return prob





def computeFirstLowerBound(CandidatePathsList, indexofCandidateLength):

    Pd = CandidatePathsList[indexofCandidateLength]



    probabilityPathExists = _path_exist_probability(Pd)



    ProbabilitySmallerPathsDontExist = 1.0

    for i in range(len(CandidatePathsList)):

        if not _is_competitor_of(CandidatePathsList, i, indexofCandidateLength):

            continue

        Pi = CandidatePathsList[i]

        private_i = _private_edges(Pi, Pd)

        exclusive_prod = _event_probability(private_i)

        ProbabilitySmallerPathsDontExist *= (1.0 - exclusive_prod)



    return ProbabilitySmallerPathsDontExist * probabilityPathExists





def LubyKarpSampling(CandidatePathsList, LKRunNumber, indexofCandidateLength):

    if LKRunNumber <= 0:

        return 0.0



    Pd = CandidatePathsList[indexofCandidateLength]

    competitor_indices = [

        i for i in range(len(CandidatePathsList))

        if _is_competitor_of(CandidatePathsList, i, indexofCandidateLength)

    ]

    competitor_paths = [CandidatePathsList[i] for i in competitor_indices]





    p_exist_d = _path_exist_probability(Pd)





    if len(competitor_paths) == 0:

        return p_exist_d





    private_events = []

    weights = []



    for Pi in competitor_paths:

        private_i = _private_edges(Pi, Pd)





        if len(private_i) == 0:

            return 0.0



        w_i = _event_probability(private_i)





        if w_i > 0.0:

            private_events.append(private_i)

            weights.append(w_i)





    if len(weights) == 0:

        return p_exist_d



    S = sum(weights)

    if S == 0.0:

        return p_exist_d





    relevant_edges = {}

    for private_i in private_events:

        for e in private_i:

            key = _edge_key(e)

            relevant_edges[key] = max(relevant_edges.get(key, 0.0), e[3])



    total = 0.0



    for _ in range(LKRunNumber):



        r = random.random() * S

        cumulative = 0.0

        chosen_idx = len(weights) - 1

        for i, w in enumerate(weights):

            cumulative += w

            if r <= cumulative:

                chosen_idx = i

                break



        chosen_private = private_events[chosen_idx]





        forced_keys = {_edge_key(e) for e in chosen_private}

        realized = {}



        for key, p in relevant_edges.items():

            if key in forced_keys:

                realized[key] = True

            else:

                realized[key] = (random.random() < p)





        N = 0

        for private_i in private_events:

            if all(realized[_edge_key(e)] for e in private_i):

                N += 1





        if N == 0:

            continue





        total += (S / float(N))



    q_hat = total / float(LKRunNumber)





    if q_hat < 0.0:

        q_hat = 0.0

    elif q_hat > 1.0:

        q_hat = 1.0



    p_sp_hat = p_exist_d * (1.0 - q_hat)



    if p_sp_hat < 0.0:

        p_sp_hat = 0.0

    elif p_sp_hat > 1.0:

        p_sp_hat = 1.0



    return p_sp_hat





def computeUpperBound(LowerBoundList, indexofCandidateLength):

    def as_scalar(x):

        return float(x[0]) if isinstance(x, (list, tuple)) else float(x)



    lower_sum = 0.0

    for i in range(indexofCandidateLength):

        lower_sum += as_scalar(LowerBoundList[i])



    upperBound = 1.0 - lower_sum



    if upperBound < 0.0:

        upperBound = 0.0

    elif upperBound > 1.0:

        upperBound = 1.0



    return upperBound

