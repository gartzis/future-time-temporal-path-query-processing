import binaryOperatorsForLearningEdgeFeatures as BOLEF

import networkx as nx

import math

import random





def _make_record(d, T, p, nodes, edges):

    return {

        "d": d,

        "T": T,

        "p": p,

        "nodes": list(nodes),

        "edges": list(edges),

    }





def _convert_L_state_to_full_records(L_dict, actual_dict):

    """
    Minimal compatibility conversion.

    Old format:
        L_dict[v] = [(d, T, p), ...]
        actual_dict[v] = (nodes, edges)

    New format expected by path_Prediction_Algorithms.py:
        L_dict[v] = [
            {"d":..., "T":..., "p":..., "nodes":[...], "edges":[...]},
            ...
        ]

    IMPORTANT:
    If a node has multiple old records but only one stored path in actual_dict[v],
    this assigns the same path to all records of that node.
    This is enough to fix the crash and make the pipeline run.
    """

    full_L = {}



    for v, records in L_dict.items():

        nodes, edges = actual_dict.get(v, ([], []))

        full_records = []



        for rec in records:

            if isinstance(rec, dict):

                full_records.append(rec)

            elif isinstance(rec, tuple) and len(rec) == 3:

                d, T, p = rec

                full_records.append(_make_record(d, T, p, nodes, edges))



        full_L[v] = full_records



    return full_L





def _prune_skyline_records(Lv):

    """
    Current/observed-state skyline:
    probability does not matter because observed edges are certain (p=1).

    Dominance:
    r1 dominates r2 iff it is no worse in distance and time,
    and strictly better in at least one.
    Exact duplicate temporal paths are collapsed.
    """

    best = {}

    for rec in Lv:

        edge_sig = tuple((a, b, tt) for (a, b, tt, _) in rec["edges"])

        key = (rec["d"], rec["T"], edge_sig)

        best[key] = rec



    items = list(best.values())



    sky = []

    for rec in items:

        d, T = rec["d"], rec["T"]

        dominated = any(

            (other["d"] <= d and other["T"] <= T) and

            (other["d"] < d or other["T"] < T)

            for other in items

        )

        if not dominated:

            sky.append(rec)



    sky.sort(key=lambda r: (r["d"], r["T"]))

    return sky



def computeEarliestArivalTime(sortedEdgeStream,vertex,vertexList,timeInterval):

    time_dict = {}


    pathList = []

    time_dict.update({vertex:timeInterval[0]})


    for v in vertexList:

        if not(v==vertex):

            time_dict.update({v:math.inf})


    for incoming_edge in sortedEdgeStream:

        prev_vertex = incoming_edge[0]

        prev_time = time_dict.get(prev_vertex)

        if incoming_edge[2]<=timeInterval[1] and incoming_edge[2]>prev_time:

            current_vertex = incoming_edge[1]

            current_time = time_dict.get(current_vertex)


            pathList.append(current_vertex)


            if incoming_edge[2]<current_time:

                time_dict.update({current_vertex:incoming_edge[2]})



        elif (incoming_edge[2]>timeInterval[1]):

            break

    return time_dict, pathList



def computeLatestDepartureTime(sortedEdgeStream,vertex,vertexList,timeInterval):

    time_dict = {}

    time_dict.update({vertex:timeInterval[1]})

    for v in vertexList:

        if not(v==vertex):

            time_dict.update({v:-math.inf})

    for edge in sortedEdgeStream:

        if edge[2]>= timeInterval[0]:



            current_vertex = edge[1]

            current_time = time_dict.get(current_vertex)

            if edge[2]<= current_time:

                prev_vertex = edge[0]

                prev_time = time_dict.get(prev_vertex)

                if edge[2]>prev_time:

                    time_dict.update({prev_vertex:edge[2]})

        else:

            break

    return time_dict



def computeFastestPathDuration(sortedEdgeStream,vertex,vertexList,timeInterval):

    fastestPath_dict = {}

    fastestPath_dict.update({vertex:0})

    S = []

    for v in vertexList:

        if not(v==vertex):

            fastestPath_dict.update({v:math.inf})

    for edge in sortedEdgeStream:



            if (edge[0]== vertex):

                if edge[2]>= timeInterval[0] and edge[2]<=timeInterval[1]:

                    S.append(edge[2])


    for time in S:

        timeInterval1 = [time,timeInterval[1]]

        time_dict, pathList = computeEarliestArivalTime(sortedEdgeStream,vertex,vertexList,timeInterval1)

        for key in list(time_dict.keys()):

            time1 = time_dict.get(key)



            fp = fastestPath_dict.get(key)

            timeDiff = time1 - time

            upValue = min(fp,timeDiff)

            fastestPath_dict.update({key:upValue})

    return fastestPath_dict



def computeFastestPathDurationOnePass(sortedEdgeStream,vertex,vertexList,timeInterval):

    vertexToLv_dict ={}

    startingTime_dict = {}

    arrivalTime_dict = {}

    fastestPath_dict = {}

    startTime_dict = {}

    fastestPath_dict.update({vertex:0})



    for v in vertexList:

        if not(v==vertex):

            fastestPath_dict.update({v:math.inf})





    for v in vertexList:

        vertexToLv_dict.update({v:[]})

        startingTime_dict.update({v:[]})

        arrivalTime_dict.update({v:[]})

        startTime_dict.update({v:[]})

    for edge in sortedEdgeStream:

        startingTimeList = startingTime_dict.get(edge[0])

        arrivalTimeList = arrivalTime_dict.get(edge[1])

        startingTimeList.append(edge[2])

        arrivalTimeList.append(edge[2])

        startingTime_dict.update({edge[0]:startingTimeList})

        arrivalTime_dict.update({edge[1]:arrivalTimeList})



    for v in vertexList:

        Lv = vertexToLv_dict.get(v)

        startTimeList = startingTime_dict.get(v)

        sTList =startTime_dict.get(v)

        arrivalTimeList = arrivalTime_dict.get(v)

        for time1 in startTimeList:

            for time2 in arrivalTimeList:

                if time1<= time2:

                    if time1 not in sTList:

                        sTList.append(time1)

                    Lv.append((time1,time2))

        Lv = sorted(Lv, key=lambda x: float(x[1]))

        startTime_dict.update({v:sTList})

        vertexToLv_dict.update({v:Lv})



    for edge in sortedEdgeStream:

        if edge[2]>= timeInterval[0] and edge[2]<= timeInterval[1]:

            Lv = vertexToLv_dict.get(edge[0])

            if edge[0] == vertex:

                if (edge[2],edge[2]) not in Lv:

                    Lv.append((edge[2],edge[2]))

                    Lv = sorted(Lv, key=lambda x: float(x[1]))

            max_atime = 0

            chosen_stime =0

            for stime,atime in Lv:

                if atime> max_atime and atime<=edge[2]:

                    chosen_stime = stime

                    max_atime = atime

            arrivalTime = edge[2]

            Lv = vertexToLv_dict.get(edge[1])

            flag = True

            for i in range(len(Lv)):

                timeSet = Lv[i]

                if timeSet[0] == chosen_stime:

                    Lv[i] = (chosen_stime,arrivalTime)

                    Lv = sorted(Lv, key=lambda x: float(x[1]))

                    flag = False

                    break

            if (flag):

                Lv.appen((chosen_stime,arrivalTime))

                Lv = sorted(Lv, key=lambda x: float(x[1]))

            dominatedElementsList = []

            for i in range(len(Lv)-1):

                s1,a1 = Lv[i]

                s2,a2 = Lv[i+1]

                if (s1>s2 and  a1 <= a2) or (s1 == s2 and a1 < a2):

                    dominatedElementsList.append(Lv[i+1])

                elif (s1<s2 and  a1 >= a2) or (s1 == s2 and a1 > a2):

                    dominatedElementsList.append(Lv[i])

            for element in dominatedElementsList:

                if element in Lv:

                    Lv.remove(element)

            fp = fastestPath_dict.get(edge[1])

            if arrivalTime - chosen_stime < fp:

                fp = arrivalTime - chosen_stime

                fastestPath_dict.update({edge[1]:fp})

        elif edge[2] > timeInterval[1]:

            break

    return  fastestPath_dict





def computeShortestPathDistance(sortedEdgeStream,vertex,vertexList,timeInterval):

    vertexToLv_dict ={}

    startingTime_dict = {}

    arrivalTime_dict = {}

    shortestPath_dict = {}

    startTime_dict = {}

    shortestPath_dict.update({vertex:0})



    for v in vertexList:

        if not(v==vertex):

            shortestPath_dict.update({v:math.inf})

        vertexToLv_dict.update({v:[]})



    '''for edge in sortedEdgeStream:
        if edge[1] in vertexList:
            if edge[0] == vertex:
                Lv = vertexToLv_dict.get(edge[0])
                Lv.append((0,edge[2]))
                Lv = sorted(Lv, key=lambda x: float(x[0]))
                vertexToLv_dict.update({edge[0]:Lv})'''

    foundEdge = False

    for edge in sortedEdgeStream:


        if edge[0] in vertexList and edge[1] in vertexList:

            if edge[0] == vertex:

                foundEdge = True

        if foundEdge == True:



            if edge[2]>= timeInterval[0] and edge[2]< timeInterval[1]:





                if edge[0] == vertex:

                    Lv = vertexToLv_dict.get(edge[0])

                    if (0,edge[2]) not in Lv:

                        Lv.append((0,edge[2]))

                        Lv = sorted(Lv, key=lambda x: float(x[0]))

                        vertexToLv_dict.update({edge[0]:Lv})

                max_atime = 0

                chosen_distance =0



                Lv = vertexToLv_dict.get(edge[0])

                for distance,atime in Lv:


                    if atime> max_atime and atime<=edge[2]:

                        chosen_distance = distance



                        max_atime = atime




                chosen_distance += 1

                atime = edge[2]



                Lv = vertexToLv_dict.get(edge[1])


                flag = True

                for i in range(len(Lv)):

                    timeSet = Lv[i]

                    if timeSet[1] == atime:


                        Lv[i] = (chosen_distance,atime)

                        Lv = sorted(Lv, key=lambda x: float(x[0]))

                        flag = False

                        break


                if (flag):

                    Lv.append((chosen_distance,atime))

                    Lv = sorted(Lv, key=lambda x: float(x[0]))

                dominatedElementsList = []

                for i in range(len(Lv)-1):

                    s1,a1 = Lv[i]

                    s2,a2 = Lv[i+1]

                    if (s1<s2 and  a1 <= a2) or (s1 == s2 and a1 < a2):

                        if Lv[i+1] not in dominatedElementsList:

                            dominatedElementsList.append(Lv[i+1])

                    elif (s1>s2 and  a1 >= a2) or (s1 == s2 and a1 > a2):

                        if Lv[i] not in dominatedElementsList:

                            dominatedElementsList.append(Lv[i])


                for element in dominatedElementsList:

                    if element in Lv:

                        Lv.remove(element)

                vertexToLv_dict.update({edge[1]:Lv})



                sp = shortestPath_dict.get(edge[1])

                if (sp >chosen_distance):


                    shortestPath_dict.update({edge[1]:chosen_distance})

            elif edge[2] > timeInterval[1]:

                break

    return shortestPath_dict






def old_computeActualShortestPathAndDistance(sortedEdgeStream,vertex,vertexList,timeInterval):


    vertexToLv_dict ={}

    startingTime_dict = {}

    arrivalTime_dict = {}

    shortestPath_dict = {}

    startTime_dict = {}

    actualPath_dict = {}

    expectedDistance_dict = {}

    shortestPath_dict.update({vertex:(0,[],0,0)})

    actualPath_dict.update({vertex:([vertex],[])})



    for v in vertexList:


        if not(v==vertex):

            shortestPath_dict.update({v:(math.inf,[],0,0)})

        actualPath_dict.update({v:([],[])})

        vertexToLv_dict.update({v:[]})

        expectedDistance_dict.update({v:[]})


    '''print('shortestPath_dict')
    print(shortestPath_dict)'''

    for edge in sortedEdgeStream:

        if edge[1] in vertexList:

            if edge[0] == vertex:

                Lv = vertexToLv_dict.get(edge[1])

                Lv.append((1,edge[2],1))


                Lv = sorted(Lv, key=lambda x: float(x[0]))

                vertexToLv_dict.update({edge[1]:Lv})

    foundEdge = False



    prevTime_dict = {}



    visitedList = []


    for edge in sortedEdgeStream:




        if edge[0] == vertex:

            foundEdge = True

        if foundEdge == True:


            if edge[0] in vertexList and edge[1] in vertexList:


                if edge[0] in prevTime_dict:

                    prevTimeList1 = prevTime_dict.get(edge[0])

                    prevTimeList = prevTimeList1.copy()


                else:

                    prevTimeList = []

                    prevTime_dict.update({edge[0]:[]})

                '''print((edge[2],edge[0]))
                print(prevTimeList)'''

                if edge[2]>= timeInterval[0] and edge[2]<= timeInterval[1] and ((edge[2],edge[0]) not in prevTimeList):


                    prevTimeList.append((edge[2],edge[1]))


                    prevTime_dict.update({edge[0]:prevTimeList})







                    if edge[0] == vertex:

                        Lv = vertexToLv_dict.get(edge[0])

                        if (0,edge[2],edge[3]) not in Lv:

                            Lv.append((0,edge[2],edge[3]))

                            Lv = sorted(Lv, key=lambda x: float(x[0]))

                            vertexToLv_dict.update({edge[0]:Lv})

                    max_atime = 0

                    chosen_distance =0

                    max_edgeProba = 1





                    Lv = vertexToLv_dict.get(edge[0])

                    '''print('Lv')
                    print(Lv)'''

                    for distance,atime,edge_proba in Lv:

                        if atime> max_atime and atime<edge[2] and max_edgeProba<=edge_proba:

                            chosen_distance = distance


                            max_atime = atime

                            max_edgeProba = edge_proba








                    chosen_distance += 1

                    atime = edge[2]

                    edge_proba = max_edgeProba*edge[3]

                    '''print(edge)
                    print(edge_proba)'''

                    '''if edge[1] not in maxPath and len(maxPath)<=chosen_distance:
                        maxPath.append(edge[1])
                    print(maxPath)'''




                    '''print(edge)
                    print('###########')
                    print(actualPath_dict)
                    print('-----------')'''

                    path,path1 = actualPath_dict.get(edge[0])

                    maxPath = path.copy()

                    new_maxPath = path.copy()

                    edge_maxPath = path1.copy()

                    '''if edge[1] == 16:
                        print(actualPath_dict)
                        print(edge)
                        print('Before:')
                        print(maxPath)'''







                    if len(new_maxPath)>0:


                        pathCheck = int(new_maxPath[0]) == int(vertex)


                        checkTime = edge[2]>path1[len(path1)-1][2]



                    else:

                        pathCheck = edge[0] == vertex

                        checkTime = True






                    if checkTime and pathCheck and edge[1] not in new_maxPath and edge[1] != vertex:





                        Lv = vertexToLv_dict.get(edge[1])

                        flag = True

                        for i in range(len(Lv)):

                            timeSet = Lv[i]

                            if timeSet[1] == atime:

                                Lv[i] = (chosen_distance,atime,edge_proba)

                                Lv = sorted(Lv, key=lambda x: float(x[0]))

                                flag = False

                                break

                        if (flag):

                            Lv.append((chosen_distance,atime,edge_proba))

                            Lv = sorted(Lv, key=lambda x: float(x[0]))



                        pathDistanceList = expectedDistance_dict.get(edge[1])



                        dominatedElementsList = []

                        for i in range(len(Lv)-1):

                            '''print('Lv[i]')
                            print(Lv[i])'''

                            s1,a1,prob1 = Lv[i]

                            s2,a2,prob2 = Lv[i+1]

                            if (s1<s2 and  a1 <= a2) or (s1 == s2 and a1 < a2):

                                if Lv[i+1] not in dominatedElementsList:

                                    dominatedElementsList.append(Lv[i+1])

                            elif (s1>s2 and  a1 >= a2) or (s1 == s2 and a1 > a2):

                                if Lv[i] not in dominatedElementsList:

                                    dominatedElementsList.append(Lv[i])

                        for element in dominatedElementsList:

                            if element in Lv:

                                Lv.remove(element)

                        '''if vertex == edge[0]:
                            print(edge)
                            print(new_maxPath)'''

                        visitedList.append((vertex,edge[1]))

                        if edge[0] not in new_maxPath:

                            new_maxPath.append(edge[0])

                        if edge[1] not in new_maxPath:

                            new_maxPath.append(edge[1])

                        if edge not in edge_maxPath:

                            edge_maxPath.append(edge)

                        '''if vertex == edge[0]:
                            print(new_maxPath)'''

                        '''if len(path1)>0:
                            checkTime = edge[2]>path1[len(path1)-1][2]
                        else:
                            checkTime = True'''



                        actualPath_dict.update({edge[1]:(new_maxPath,edge_maxPath)})

                    '''if len(new_maxPath) > chosen_distance:
                        new_maxPath = maxPath.copy()
                        if edge[1] not in new_maxPath:
                            new_maxPath.append(edge[1])'''

                    '''if edge[1] ==16:
                        print('After:')
                        print(maxPath)
                        print('-------')'''












                    vertexToLv_dict.update({edge[1]:Lv})


                    sp = shortestPath_dict.get(edge[1])



                    sp = sp[0]





                    maxPath1,edge_maxPath = actualPath_dict.get(edge[1])

                    chosen_distance = len(maxPath1)-1

                    if (sp >chosen_distance) and vertex in maxPath1:




                            if len(maxPath1)>0 and vertex in maxPath1:


                                flag = True

                                for i in range(len(pathDistanceList)):

                                    timeInstance = pathDistanceList[i][1]

                                    prev_distance = pathDistanceList[i][0]

                                    if timeInstance == atime and prev_distance>chosen_distance:

                                        pathDistanceList[i] = (chosen_distance,atime,maxPath1,edge_maxPath,edge_proba)

                                        flag = False

                                        break

                                if (flag):

                                    pathDistanceList.append((chosen_distance,atime,maxPath1,edge_maxPath,edge_proba))

                                pathDistanceList = sorted(pathDistanceList, key=lambda x: float(x[0]))

                                expectedDistance_dict.update({edge[1]:pathDistanceList})

                                '''if vertex not in maxPath1:
                                    maxPath1 = []
                                    chosen_distance = math.inf'''

                                if len(maxPath1)-1>chosen_distance:

                                    chosen_distance = len(maxPath1)-1

                                if len(maxPath1) == 0:

                                        chosen_distance = math.inf







                            shortestPath_dict.update({edge[1]:(chosen_distance,maxPath1,edge[2],1)})

                elif edge[2] > timeInterval[1]:

                    break



        shortestPath_dict.update({vertex:(0,[vertex],1,1)})

    vertexToLv_dict = _convert_L_state_to_full_records(vertexToLv_dict, actualPath_dict)



    return shortestPath_dict, vertexToLv_dict, shortestPath_dict, actualPath_dict, expectedDistance_dict








def computeActualShortestPathAndDistanceProba(sortedEdgeStream,vertex,vertexList,timeInterval):

    vertexToLv_dict ={}

    startingTime_dict = {}

    arrivalTime_dict = {}

    shortestPath_dict = {}

    startTime_dict = {}

    actualPath_dict = {}

    shortestPath_dict.update({vertex:(0,[])})

    actualPath_dict.update({vertex:[vertex]})

    for v in vertexList:


        if not(v==vertex):

            shortestPath_dict.update({v:(math.inf,[])})

        actualPath_dict.update({v:[]})

        vertexToLv_dict.update({v:[]})


    for edge in sortedEdgeStream:

        if edge[1] in vertexList:

            if edge[0] == vertex:

                Lv = vertexToLv_dict.get(edge[1])

                Lv.append((1,edge[2],1))


                Lv = sorted(Lv, key=lambda x: float(x[0]))

                vertexToLv_dict.update({edge[1]:Lv})

    foundEdge = False



    prevTime_dict = {}



    visitedList = []

    for edge in sortedEdgeStream:


        if edge[0] == vertex:

            foundEdge = True

        if foundEdge == True:

            if edge[0] in vertexList and edge[1] in vertexList:

                if edge[0] in prevTime_dict:

                    prevTimeList = prevTime_dict.get(edge[0])


                else:

                    prevTimeList = []

                    prevTime_dict.update({edge[0]:[]})

                if edge[2]>= timeInterval[0] and edge[2]< timeInterval[1] and ((edge[2],edge[0]) not in prevTimeList):

                    prevTimeList.append((edge[2],edge[1]))


                    prevTime_dict.update({edge[0]:prevTimeList})







                    if edge[0] == vertex:

                        Lv = vertexToLv_dict.get(edge[0])

                        if (0,edge[2],edge[3]) not in Lv:

                            Lv.append((0,edge[2],edge[3]))

                            Lv = sorted(Lv, key=lambda x: float(x[0]))

                            vertexToLv_dict.update({edge[0]:Lv})

                    max_atime = 0

                    chosen_distance =0

                    max_edgeProba = 1





                    Lv = vertexToLv_dict.get(edge[0])

                    for distance,atime,edge_proba in Lv:

                        if atime> max_atime and atime<edge[2] and max_edgeProba<=edge_proba:

                            chosen_distance = distance


                            max_atime = atime

                            max_edgeProba = edge_proba






                    chosen_distance += 1

                    atime = edge[2]

                    edge_proba = max_edgeProba*edge[3]


                    '''if edge[1] not in maxPath and len(maxPath)<=chosen_distance:
                        maxPath.append(edge[1])
                    print(maxPath)'''

                    Lv = vertexToLv_dict.get(edge[1])

                    flag = True

                    for i in range(len(Lv)):

                        timeSet = Lv[i]

                        if timeSet[1] == atime:

                            Lv[i] = (chosen_distance,atime,edge_proba)

                            Lv = sorted(Lv, key=lambda x: float(x[0]))

                            flag = False

                            break

                    if (flag):

                        Lv.append((chosen_distance,atime,edge_proba))

                        Lv = sorted(Lv, key=lambda x: float(x[0]))

                    dominatedElementsList = []

                    for i in range(len(Lv)-1):

                        '''print('Lv[i]')
                        print(Lv[i])'''

                        s1,a1,prob1 = Lv[i]

                        s2,a2,prob2 = Lv[i+1]

                        '''if prob1 ==1:
                            prob1 = -1
                        if prob2 ==1:
                            prob1 = -1'''


                        if (s1<s2 and  a1 <= a2) or (s1 == s2 and a1 < a2) and prob1>=0.5:

                            if Lv[i+1] not in dominatedElementsList:

                                dominatedElementsList.append(Lv[i+1])


                        elif (s1>s2 and  a1 >= a2) or (s1 == s2 and a1 > a2) and 0.5<=prob2:

                            if Lv[i] not in dominatedElementsList:

                                dominatedElementsList.append(Lv[i])

                    for element in dominatedElementsList:

                        if element in Lv:

                            Lv.remove(element)


                    '''print(edge)
                    print('###########')
                    print(actualPath_dict)
                    print('-----------')'''

                    path = actualPath_dict.get(edge[0])

                    maxPath = path.copy()

                    new_maxPath = path.copy()

                    '''if edge[1] == 16:
                        print(actualPath_dict)
                        print(edge)
                        print('Before:')
                        print(maxPath)'''









                    if ((vertex,edge[1]) not in visitedList or edge[0]==vertex) and edge[1]!=vertex:

                        '''if vertex == edge[0]:
                            print(edge)
                            print(new_maxPath)'''

                        visitedList.append((vertex,edge[1]))

                        if edge[0] not in new_maxPath:

                            new_maxPath.append(edge[0])

                        if edge[1] not in new_maxPath:

                            new_maxPath.append(edge[1])

                        '''if vertex == edge[0]:
                            print(new_maxPath)'''

                        actualPath_dict.update({edge[1]:new_maxPath})

                    '''if len(new_maxPath) > chosen_distance:
                        new_maxPath = maxPath.copy()
                        if edge[1] not in new_maxPath:
                            new_maxPath.append(edge[1])'''

                    '''if edge[1] ==16:
                        print('After:')
                        print(maxPath)
                        print('-------')'''












                    vertexToLv_dict.update({edge[1]:Lv})




                    sp = shortestPath_dict.get(edge[1])



                    sp = sp[0]





                    if (sp >chosen_distance):


                            maxPath1 = actualPath_dict.get(edge[1])

                            if vertex not in maxPath1:

                                maxPath1 = []

                                chosen_distance = math.inf

                            if len(maxPath1)-1>chosen_distance:

                                chosen_distance = len(maxPath1)-1





                            shortestPath_dict.update({edge[1]:(chosen_distance,maxPath1)})

                elif edge[2] > timeInterval[1]:

                    break


        shortestPath_dict.update({vertex:(0,[vertex])})


    '''print('vertexToLv_dict')
    print(vertexToLv_dict)'''

    return shortestPath_dict,vertexToLv_dict,shortestPath_dict,actualPath_dict,



def predictActualShortestPathAndDistanceProba(sortedEdgeStream,vertexToLv_dict,shortestPath_dict,actualPath_dict,vertex,vertexList,timeInterval):


    '''startingTime_dict = {}
    arrivalTime_dict = {}
    shortestPath_dict = {}
    startTime_dict = {}
    actualPath_dict = {}
    shortestPath_dict.update({vertex:(0,[])})
    actualPath_dict.update({vertex:[vertex]})'''

    '''for v in vertexList:
        if not(v==vertex):
            shortestPath_dict.update({v:(math.inf,[])})
        actualPath_dict.update({v:[]})
        vertexToLv_dict.update({v:[]})'''


    '''for edge in sortedEdgeStream:
        if edge[1] in vertexList:
            if edge[0] not in vertexToLv_dict.keys():
                if edge[0] == vertex:
                    Lv = vertexToLv_dict.get(edge[1])
                    Lv.append((1,edge[2],1))
                    Lv = sorted(Lv, key=lambda x: float(x[0]))
                    vertexToLv_dict.update({edge[1]:Lv})'''

    foundEdge = False



    prevTime_dict = {}



    visitedList = []

    print(sortedEdgeStream)

    for edge in sortedEdgeStream:

        print(edge)


        foundEdge = True

        if foundEdge == True:

            if edge[0] in vertexList and edge[1] in vertexList:

                print(edge)

                if edge[0] in prevTime_dict:

                    prevTimeList1 = prevTime_dict.get(edge[0])

                    prevTimeList = prevTimeList1.copy()


                else:

                    prevTimeList = []

                    prevTime_dict.update({edge[0]:[]})

                if edge[2]>= timeInterval[0] and edge[2]< timeInterval[1] and ((edge[2],edge[0]) not in prevTimeList):

                    print(edge)

                    prevTimeList.append((edge[2],edge[1]))


                    prevTime_dict.update({edge[0]:prevTimeList})







                    if edge[0] == vertex:

                        Lv = vertexToLv_dict.get(edge[0])

                        if (0,edge[2],edge[3]) not in Lv:

                            Lv.append((0,edge[2],edge[3]))

                            Lv = sorted(Lv, key=lambda x: float(x[0]))

                            vertexToLv_dict.update({edge[0]:Lv})

                    max_atime = 0

                    chosen_distance =0

                    max_edgeProba = 0.5





                    Lv = vertexToLv_dict.get(edge[0])


                    Lv_prev = vertexToLv_dict.get(edge[1])

                    if len(Lv_prev)>0:

                        prevpathProba = Lv_prev[0][2]

                    else:

                        prevpathProba = 0

                    '''print('Lv')
                    print(Lv)'''

                    for distance,atime,edge_proba in Lv:

                        if atime> max_atime and atime<edge[2] and max_edgeProba<=edge_proba:

                            chosen_distance = distance


                            max_atime = atime

                            max_edgeProba = edge_proba






                    chosen_distance += 1

                    atime = edge[2]


                    edge_proba = max_edgeProba*edge[3]

                    newPathProba = edge_proba






                    '''if edge[1] not in maxPath and len(maxPath)<=chosen_distance:
                        maxPath.append(edge[1])
                    print(maxPath)'''

                    Lv = vertexToLv_dict.get(edge[1])

                    numOfPaths = len(Lv)

                    flag = True

                    for i in range(len(Lv)):

                        timeSet = Lv[i]

                        if timeSet[1] == atime:

                            Lv[i] = (chosen_distance,atime,edge_proba)

                            Lv = sorted(Lv, key=lambda x: float(x[0]))

                            flag = False

                            break

                    if (flag):

                        Lv.append((chosen_distance,atime,edge_proba))

                        Lv = sorted(Lv, key=lambda x: float(x[0]))

                    dominatedElementsList = []





                    '''while len(Lv)>1:

                        i = 0'''

                    for i in range(len(Lv)-1):

                        '''print('Lv[i]')
                        print(Lv[i])'''

                        s1,a1,prob1 = Lv[i]


                        s2,a2,prob2 = Lv[i+1]


                        '''if prob1 ==1:
                            prob1 = -1
                        if prob2 ==1:
                            prob1 = -1'''



                        if prob2>prob1 and prob2 !=1:

                            if Lv[i] not in dominatedElementsList:

                                dominatedElementsList.append(Lv[i])



                        else:

                            if Lv[i+1] not in dominatedElementsList:

                                    dominatedElementsList.append(Lv[i+1])




                        '''else:
                        if ((s1<s2 and  a1 <= a2) or (s1 == s2 and a1 < a2)) or prob1>=0.5:
                                print('dominated 2:\t'+str(Lv[i+1]))
                                if Lv[i+1] not in dominatedElementsList:
                                    dominatedElementsList.append(Lv[i+1])
                        elif ((s1>s2 and  a1 >= a2) or (s1 == s2 and a1 > a2)) or prob2>=0.5:
                                print('dominated 1:\t'+str(Lv[i]))
                                if Lv[i] not in dominatedElementsList:
                                    dominatedElementsList.append(Lv[i])'''





                        for element in dominatedElementsList:

                                if element in Lv:

                                    Lv.remove(element)

                    '''print('edge[1]')
                    print(edge[1])
                    print('dominatedElementsList')
                    print(dominatedElementsList)'''







                    '''print(edge)
                    print('###########')
                    print(actualPath_dict)
                    print('-----------')'''

                    print(edge[0])

                    path = actualPath_dict.get(edge[0])

                    print(path)



                    maxPath = path.copy()

                    '''print(edge)
                    print('maxPath')
                    print(maxPath)'''

                    new_maxPath = path.copy()

                    '''if edge[1] == 16:
                        print(actualPath_dict)
                        print(edge)
                        print('Before:')
                        print(maxPath)'''









                    if ((vertex,edge[1]) not in visitedList or edge[0]==vertex) and edge[1]!=vertex:

                        print('here')

                        print(new_maxPath)

                        '''if vertex == edge[0]:
                            print(edge)
                            print(new_maxPath)'''

                        visitedList.append((vertex,edge[1]))

                        if edge[0] not in new_maxPath:

                            new_maxPath.append(edge[0])

                        if edge[1] not in new_maxPath:

                            new_maxPath.append(edge[1])

                        print(new_maxPath)

                        '''if vertex == edge[0]:
                            print(new_maxPath)'''

                        actualPath_dict.update({edge[1]:new_maxPath})

                    '''if len(new_maxPath) > chosen_distance:
                        new_maxPath = maxPath.copy()
                        if edge[1] not in new_maxPath:
                            new_maxPath.append(edge[1])'''

                    '''if edge[1] ==16:
                        print('After:')
                        print(maxPath)
                        print('-------')'''












                    vertexToLv_dict.update({edge[1]:Lv})




                    sp = shortestPath_dict.get(edge[1])



                    sp = sp[0]



                    print(chosen_distance)

                    print(sp)

                    print('prevpathProba:\t'+str(prevpathProba))

                    print('newPathProba:\t'+str(newPathProba))

                    prevShortestPathProba = prevpathProba/numOfPaths

                    newShortestPathProba = newPathProba/numOfPaths

                    print('prevShortestPathProba:\t'+str(prevShortestPathProba))

                    print('newShortestPathProba:\t'+str(newShortestPathProba))

                    if (sp >chosen_distance and (prevShortestPathProba==1 or newShortestPathProba >=prevShortestPathProba)):

                            print('here1')


                            maxPath1 = actualPath_dict.get(edge[1])

                            if vertex not in maxPath1:

                                maxPath1 = []

                                chosen_distance = math.inf

                            print(chosen_distance)

                            print(maxPath1)

                            if len(maxPath1)-1>chosen_distance:

                                chosen_distance = len(maxPath1)-1





                            shortestPath_dict.update({edge[1]:(chosen_distance,maxPath1)})

                elif edge[2] > timeInterval[1]:

                    break


        shortestPath_dict.update({vertex:(0,[vertex])})


    '''print('vertexToLv_dict')
    print(vertexToLv_dict)'''

    return shortestPath_dict,vertexToLv_dict,shortestPath_dict,actualPath_dict





def predictShortestPathDistance(clf,embeddings_dict,vertex0,sortedEdgeStream,vertexList,timeInterval,embeddingPassingType,destinationNodesNum):



    shortestPath_dict = computeShortestPathDistance(sortedEdgeStream,vertex0,vertexList,timeInterval)



    distanceNodeList_dict = {}



    for node in list(shortestPath_dict.keys()):

        if node in vertexList:

            distance = shortestPath_dict.get(node)

            if distance in list(distanceNodeList_dict.keys()):

                nodeList =distanceNodeList_dict.get(distance)

                nodeList.append(node)

                distanceNodeList_dict.update({distance:nodeList})

            else:

                nodeList = []

                nodeList.append(node)

                distanceNodeList_dict.update({distance:nodeList})

    nodesList = list(shortestPath_dict.keys())

    destinationVertexList = []

    while len(destinationVertexList)<destinationNodesNum:

        destNode = random.choice(nodesList)

        if destNode not in destinationVertexList and not(destNode == vertex0):

            destinationVertexList.append(destNode)




    for vertex in destinationVertexList:

        distanceNodeKeysList = list(distanceNodeList_dict.keys())

        distanceNodeKeysList.sort()



        node1_distance = shortestPath_dict.get(vertex)

        if (node1_distance == math.inf):

            node1_distance = distanceNodeKeysList[len(distanceNodeKeysList)-2]+1



        for distance in range(0,node1_distance-1):



            nodeList = distanceNodeList_dict.get(distance)

            for node in nodeList:

                edge = (node,vertex)

                if embeddingPassingType == 'Hadamard':

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeHadamard(edge,embeddings_dict)

                elif embeddingPassingType == 'WeightedL1':

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL1(edge,embeddings_dict)

                elif embeddingPassingType == 'WeightedL2':

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL2(edge,embeddings_dict)

                else:

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeAverage(edge,embeddings_dict)

                predictionResult = clf.predict([embeddingEdgeRepresentation])

                predictionResultProba = clf.predict_proba([embeddingEdgeRepresentation])

                '''print("predictionResult:\t"+str(predictionResult))
                print("predictionResultProba:\t"+str(predictionResultProba))'''

                if predictionResult == 1:

                    if node1_distance == distanceNodeKeysList[len(distanceNodeKeysList)-2]+1:

                        node1_distance = math.inf



                    if node1_distance in list(distanceNodeList_dict.keys()):

                        nodeList1 = distanceNodeList_dict.get(node1_distance)

                        if vertex in nodeList1:

                            nodeList1.remove(vertex)

                        distanceNodeList_dict.update({node1_distance:nodeList1})

                        node1_distance = distance + 1

                        shortestPath_dict.update({vertex:node1_distance})


                        nodeList1 = distanceNodeList_dict.get(node1_distance)

                        if vertex not in nodeList1:

                            nodeList1.append(vertex)

                            distanceNodeList_dict.update({node1_distance:nodeList1})

                        break

    return shortestPath_dict



def predictShortestPathDistance(clf,embeddings_dict,vertex0,sortedEdgeStream,vertexList,timeInterval,embeddingPassingType,destinationNodesNum):



    shortestPath_dict = computeShortestPathDistance(sortedEdgeStream,vertex0,vertexList,timeInterval)



    distanceNodeList_dict = {}



    for node in list(shortestPath_dict.keys()):

        if node in vertexList:

            distance = shortestPath_dict.get(node)

            if distance in list(distanceNodeList_dict.keys()):

                nodeList =distanceNodeList_dict.get(distance)

                nodeList.append(node)

                distanceNodeList_dict.update({distance:nodeList})

            else:

                nodeList = []

                nodeList.append(node)

                distanceNodeList_dict.update({distance:nodeList})

    nodesList = list(shortestPath_dict.keys())

    destinationVertexList = []

    while len(destinationVertexList)<destinationNodesNum:

        destNode = random.choice(nodesList)

        if destNode not in destinationVertexList and not(destNode == vertex0):

            destinationVertexList.append(destNode)




    for vertex in destinationVertexList:

        distanceNodeKeysList = list(distanceNodeList_dict.keys())

        distanceNodeKeysList.sort()



        node1_distance = shortestPath_dict.get(vertex)

        if (node1_distance == math.inf):

            node1_distance = distanceNodeKeysList[len(distanceNodeKeysList)-2]+1



        for distance in range(0,node1_distance-1):



            nodeList = distanceNodeList_dict.get(distance)

            for node in nodeList:

                edge = (node,vertex)

                if embeddingPassingType == 'Hadamard':

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeHadamard(edge,embeddings_dict)

                elif embeddingPassingType == 'WeightedL1':

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL1(edge,embeddings_dict)

                elif embeddingPassingType == 'WeightedL2':

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL2(edge,embeddings_dict)

                else:

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeAverage(edge,embeddings_dict)

                predictionResult = clf.predict([embeddingEdgeRepresentation])

                predictionResultProba = clf.predict_proba([embeddingEdgeRepresentation])

                '''print("predictionResult:\t"+str(predictionResult))
                print("predictionResultProba:\t"+str(predictionResultProba))'''

                if predictionResult == 1:

                    if node1_distance == distanceNodeKeysList[len(distanceNodeKeysList)-2]+1:

                        node1_distance = math.inf



                    if node1_distance in list(distanceNodeList_dict.keys()):

                        nodeList1 = distanceNodeList_dict.get(node1_distance)

                        if vertex in nodeList1:

                            nodeList1.remove(vertex)

                        distanceNodeList_dict.update({node1_distance:nodeList1})

                        node1_distance = distance + 1

                        shortestPath_dict.update({vertex:node1_distance})


                        nodeList1 = distanceNodeList_dict.get(node1_distance)

                        if vertex not in nodeList1:

                            nodeList1.append(vertex)

                            distanceNodeList_dict.update({node1_distance:nodeList1})

                        break

    return shortestPath_dict





def predictProbaShortestPathDistance(clf,embeddings_dict,vertex0,sortedEdgeStream,vertexList,timeInterval,embeddingPassingType,destinationNodesNum):



    shortestPath_dict = computeShortestPathDistance(sortedEdgeStream,vertex0,vertexList,timeInterval)



    distanceNodeList_dict = {}



    for node in list(shortestPath_dict.keys()):

        if node in vertexList:

            distance = shortestPath_dict.get(node)

            if distance in list(distanceNodeList_dict.keys()):

                nodeList =distanceNodeList_dict.get(distance)

                nodeList.append(node)

                distanceNodeList_dict.update({distance:nodeList})

            else:

                nodeList = []

                nodeList.append(node)

                distanceNodeList_dict.update({distance:nodeList})

    nodesList = list(shortestPath_dict.keys())

    destinationVertexList = []

    while len(destinationVertexList)<destinationNodesNum:

        destNode = random.choice(nodesList)

        if destNode not in destinationVertexList and not(destNode == vertex0):

            destinationVertexList.append(destNode)




    for vertex in destinationVertexList:

        distanceNodeKeysList = list(distanceNodeList_dict.keys())

        distanceNodeKeysList.sort()

        maxPosibility = 0.5



        node1_distance = shortestPath_dict.get(vertex)

        if (node1_distance == math.inf):

            node1_distance = distanceNodeKeysList[len(distanceNodeKeysList)-2]+1



        for distance in range(0,node1_distance-1):



            nodeList = distanceNodeList_dict.get(distance)

            for node in nodeList:

                edge = (node,vertex)

                if embeddingPassingType == 'Hadamard':

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeHadamard(edge,embeddings_dict)

                elif embeddingPassingType == 'WeightedL1':

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL1(edge,embeddings_dict)

                elif embeddingPassingType == 'WeightedL2':

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL2(edge,embeddings_dict)

                else:

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeAverage(edge,embeddings_dict)

                predictionResult = clf.predict([embeddingEdgeRepresentation])

                predictionResultProba = list(clf.predict_proba([embeddingEdgeRepresentation]))[0]

                currentProba = list(predictionResultProba)[1]

                '''print("predictionResult:\t"+str(predictionResult))
                print("predictionResultProba:\t"+str(predictionResultProba))'''

                if currentProba >maxPosibility:

                    maxPosibility = currentProba

                    if node1_distance == distanceNodeKeysList[len(distanceNodeKeysList)-2]+1:

                        node1_distance = math.inf



                    if node1_distance in list(distanceNodeList_dict.keys()):

                        nodeList1 = distanceNodeList_dict.get(node1_distance)

                        if vertex in nodeList1:

                            nodeList1.remove(vertex)

                        distanceNodeList_dict.update({node1_distance:nodeList1})

                        node1_distance = distance + 1

                        shortestPath_dict.update({vertex:node1_distance})


                        nodeList1 = distanceNodeList_dict.get(node1_distance)

                        if vertex not in nodeList1:

                            nodeList1.append(vertex)

                            distanceNodeList_dict.update({node1_distance:nodeList1})

                        break

    return shortestPath_dict



def predictExpectedLengthShortestPathDistance(clf,embeddings_dict,vertex0,sortedEdgeStream,vertexList,destinatoinVertexList,timeInterval,embeddingPassingType,destinationNodesNum):



    shortestPath_dict = computeShortestPathDistance(sortedEdgeStream,vertex0,vertexList,timeInterval)



    distanceNodeList_dict = {}



    for node in list(shortestPath_dict.keys()):

        if node in vertexList:

            distance = shortestPath_dict.get(node)

            if distance in list(distanceNodeList_dict.keys()):

                nodeList =distanceNodeList_dict.get(distance)

                nodeList.append(node)

                distanceNodeList_dict.update({distance:nodeList})

            else:

                nodeList = []

                nodeList.append(node)

                distanceNodeList_dict.update({distance:nodeList})



    '''while len(destinationVertexList)<destinationNodesNum:
        destNode = random.choice(nodesList)
        if destNode not in destinationVertexList and not(destNode == vertex0):
            destinationVertexList.append(destNode)'''




    for vertex in destinatoinVertexList:

        distanceNodeKeysList = list(distanceNodeList_dict.keys())

        distanceNodeKeysList.sort()

        maxPosibility = 0.5

        averageDistanceProba_dict = {}

        node1_distance = shortestPath_dict.get(vertex)


        if (node1_distance == math.inf):

            node1_distance = distanceNodeKeysList[len(distanceNodeKeysList)-2]+1



        for distance in range(0,node1_distance):

            currentProba = 1

            nodeList = distanceNodeList_dict.get(distance)

            for node in nodeList:

                edge = (node,vertex)

                if embeddingPassingType == 'Hadamard':

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeHadamard(edge,embeddings_dict)

                elif embeddingPassingType == 'WeightedL1':

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL1(edge,embeddings_dict)

                elif embeddingPassingType == 'WeightedL2':

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeWeightedL2(edge,embeddings_dict)

                else:

                    embeddingEdgeRepresentation = BOLEF.createEmbeddingEdgeAverage(edge,embeddings_dict)

                predictionResult = clf.predict([embeddingEdgeRepresentation])

                predictionResultProba = list(clf.predict_proba([embeddingEdgeRepresentation]))[0]


                currentProba = currentProba*(1-list(predictionResultProba)[1])


                '''print("predictionResult:\t"+str(predictionResult))
                print("predictionResultProba:\t"+str(predictionResultProba))'''

                '''if currentProba >maxPosibility:
                    maxPosibility = currentProba'''

            distKey = distance+1

            currentProba = 1 - currentProba




            '''if currentProba >=1:
                currentProba = 0'''

            averageDistanceProba_dict.update({distKey:currentProba})

            if node1_distance == distanceNodeKeysList[len(distanceNodeKeysList)-2]+1:

                        node1_distance = math.inf



            '''if node1_distance in list(distanceNodeList_dict.keys()):
                        nodeList1 = distanceNodeList_dict.get(node1_distance)
                        if vertex in nodeList1:
                            nodeList1.remove(vertex)
                        distanceNodeList_dict.update({node1_distance:nodeList1})
                        node1_distance = distance + 1'''

        tempDist = 0


        '''if len(list(averageDistanceProba_dict.keys()))>0:
            print(averageDistanceProba_dict)'''

        for distKey in list(averageDistanceProba_dict.keys()):

                distProba = averageDistanceProba_dict.get(distKey)




                tempDist = tempDist + int(distKey)*distProba


        if vertex == destinatoinVertexList[1]:

            '''print('vertex:\t'+str(vertex)+'\tdistance:\t'+str(tempDist))
            print(tempDist)'''

        if tempDist !=0:

            shortestPath_dict.update({vertex:math.ceil(tempDist)})





        '''if vertex not in nodeList1:
                            nodeList1.append(vertex)
                            distanceNodeList_dict.update({node1_distance:nodeList1})'''

    return shortestPath_dict



'''graph_file_path = 'Temporal_Stream_File.txt'
graphFileRead = open(graph_file_path, "r")
streamEdgeList = [] #should be shorted based on time
vertexList = []
minTime = math.inf
maxTime = 0
for line in file_reader(graphFileRead):
    line = line.split(',')
    edge =(int(line[0]),int(line[1]),float(line[2]))
    if edge[0] not in vertexList:
         vertexList.append(edge[0])
    if edge[1] not in vertexList:
         vertexList.append(edge[1])
    if minTime> edge[2]:
        minTime = edge[2]
    if maxTime< edge[2]:
        maxTime = edge[2]
    streamEdgeList.append(edge)
timeInterval = [minTime,maxTime]
sortedEdgeStream = sorted(streamEdgeList, key=lambda x: float(x[2]))
sortedReverseEdgeStream = sorted(streamEdgeList, key=lambda x: float(x[2]),reverse=True)
print("Max Time Interval is:   "+str((minTime,maxTime)))'''

'''for i in range(len(sortedReverseEdgeStream)-1):
    edge0 = sortedReverseEdgeStream[i]
    edge1 = sortedReverseEdgeStream[i+1]
    if (edge0[2]>edge1[2]):
        print(i)'''


'''time_dict,pathList = computeEarliestArivalTime(sortedEdgeStream,vertexList[0],vertexList,timeInterval)
print(len(pathList))'''


'''for vertex in list(time_dict.keys()):
    earliest_time = time_dict.get(vertex)

    print("Earliest Arival path at vertex  "+str(vertex)+' is '+str(earliest_time))'''




'''for vertex in list(latestDepartureTime_dict.keys()):
    earliest_time = latestDepartureTime_dict.get(vertex)

    print("Latest Departure path at vertex  "+str(vertex)+' is '+str(earliest_time))'''










'''for vertex in list(fastestPath_dict.keys()):
    fastest_path = fastestPath_dict.get(vertex)

    print("Latest Departure path at vertex  "+str(vertex)+' is '+str(fastest_path))'''





'''fastestPath_dict =  computeShortestPathDistance(sortedEdgeStream,vertexList[0],vertexList,timeInterval)

for vertex in list(fastestPath_dict.keys()):
    fastest_path = fastestPath_dict.get(vertex)

    print("Shortest path at vertex  (one pass)   "+str(vertex)+' is of length '+str(fastest_path))
distanceNodeList_dict = {}
for node in range(len(list(fastestPath_dict.keys()))):
    distance = fastestPath_dict.get(node)
    if distance in list(distanceNodeList_dict.keys()):
        nodeList =distanceNodeList_dict.get(distance)
        nodeList.append(node)
        distanceNodeList_dict.update({distance:nodeList})
    else:
        nodeList = []
        nodeList.append(node)
        distanceNodeList_dict.update({distance:nodeList})'''








def _tuple_record_to_dict_record(rec):

    d, T, nodes, edges, p = rec

    return {

        "d": d,

        "T": T,

        "nodes": list(nodes),

        "edges": list(edges),

        "p": p,

        "prob": p,

    }





def _convert_L_tuple_state_to_dict_state(state_dict):

    out = {}

    for v, Lv in state_dict.items():

        out[v] = [_tuple_record_to_dict_record(rec) for rec in Lv]

    return out



def _latest_feasible_record(Lu, t_edge):

    """
    Pick the feasible predecessor record with the latest arrival time.
    Tie-breaks:
      1) larger arrival time
      2) smaller distance
      3) larger probability
    Record format:
      (distance, arrival_time, path_nodes, path_edges, path_prob)
    """

    feasible = [rec for rec in Lu if rec[1] < t_edge]

    if not feasible:

        return None

    return max(feasible, key=lambda r: (r[1], -r[0], r[4]))





def _prune_L_records(Lv):

    """
    Keep only non-dominated records.

    Dominance is based on GRADES logic:
      r1 dominates r2 if
        - r1.d < r2.d and r1.a <= r2.a, or
        - r1.d == r2.d and r1.a < r2.a

    For exact same (d, a), keep only the one with larger probability.
    """

    if not Lv:

        return []




    best_same_da = {}

    for rec in Lv:

        d, a, nodes, edges, p = rec

        key = (d, a)

        if key not in best_same_da or p > best_same_da[key][4]:

            best_same_da[key] = rec



    items = sorted(best_same_da.values(), key=lambda r: (r[0], r[1], -r[4]))



    pruned = []

    for rec in items:

        d, a, _, _, _ = rec



        dominated = False

        for kept in pruned:

            kd, ka, _, _, _ = kept

            if (kd < d and ka <= a) or (kd == d and ka < a):

                dominated = True

                break



        if dominated:

            continue




        new_pruned = []

        for kept in pruned:

            kd, ka, _, _, _ = kept

            if (d < kd and a <= ka) or (d == kd and a < ka):

                continue

            new_pruned.append(kept)



        new_pruned.append(rec)

        pruned = new_pruned



    pruned.sort(key=lambda r: (r[0], r[1], -r[4]))

    return pruned





def _best_record_for_node(Lv):

    """
    Best record for shortestPath_dict:
      1) minimum distance
      2) earliest arrival
      3) highest probability
    """

    if not Lv:

        return None

    return min(Lv, key=lambda r: (r[0], r[1], -r[4]))





def computeActualShortestPathAndDistance(sortedEdgeStream, vertex, vertexList, timeInterval):

    """
    Correct current-state temporal shortest path computation with path probability.

    L_v record format:
        (distance, arrival_time, path_nodes, path_edges, path_prob)

    shortestPath_dict[v] format:
        (best_distance, best_path_nodes, best_arrival_time, best_path_prob)

    actualPath_dict[v] format:
        (best_path_nodes, best_path_edges)

    expectedDistance_dict[v]:
        list of full non-dominated records in L_v
    """

    vertex_set = set(vertexList)



    vertexToLv_dict = {}

    shortestPath_dict = {}

    actualPath_dict = {}

    expectedDistance_dict = {}




    for v in vertexList:

        vertexToLv_dict[v] = []

        expectedDistance_dict[v] = []

        actualPath_dict[v] = ([], [])

        shortestPath_dict[v] = (math.inf, [], 0, 0.0)




    source_record = (0, timeInterval[0], [vertex], [], 1.0)

    vertexToLv_dict[vertex] = [source_record]

    expectedDistance_dict[vertex] = [source_record]

    actualPath_dict[vertex] = ([vertex], [])

    shortestPath_dict[vertex] = (0, [vertex], timeInterval[0], 1.0)




    for edge in sortedEdgeStream:

        u, v, t, edge_prob = edge



        if t < timeInterval[0]:

            continue

        if t > timeInterval[1]:

            break

        if u not in vertex_set or v not in vertex_set:

            continue




        pred = _latest_feasible_record(vertexToLv_dict[u], t)

        if pred is None:

            continue



        pred_d, pred_a, pred_nodes, pred_edges, pred_prob = pred




        if v in pred_nodes:

            continue



        new_d = pred_d + 1

        new_a = t

        new_nodes = pred_nodes + [v]

        new_edges = pred_edges + [edge]

        new_prob = pred_prob * edge_prob



        new_rec = (new_d, new_a, new_nodes, new_edges, new_prob)



        Lv = vertexToLv_dict[v]

        Lv.append(new_rec)

        Lv = _prune_L_records(Lv)

        vertexToLv_dict[v] = Lv

        expectedDistance_dict[v] = Lv.copy()



        best = _best_record_for_node(Lv)

        if best is not None:

            best_d, best_a, best_nodes, best_edges, best_prob = best

            shortestPath_dict[v] = (best_d, best_nodes, best_a, best_prob)

            actualPath_dict[v] = (best_nodes, best_edges)




    shortestPath_dict[vertex] = (0, [vertex], timeInterval[0], 1.0)

    actualPath_dict[vertex] = ([vertex], [])

    vertexToLv_dict[vertex] = [source_record]

    expectedDistance_dict[vertex] = [source_record]



    vertexToLv_dict_out = _convert_L_tuple_state_to_dict_state(vertexToLv_dict)

    expectedDistance_dict_out = _convert_L_tuple_state_to_dict_state(expectedDistance_dict)



    return (

        shortestPath_dict,

        vertexToLv_dict_out,

        shortestPath_dict,

        actualPath_dict,

        expectedDistance_dict_out,

    )
