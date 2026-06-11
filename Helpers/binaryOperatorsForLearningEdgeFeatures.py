import random

from FileReader import file_reader







def createTrainSetAverage(edgesList,nodesList,embeddings_dict,deletedEdgesList):

    embeddingEdgesList = []

    edgeExistList = []

    edgeList = []

    for edge in edgesList:

        edgeList.append(edge)

        embedding1 = embeddings_dict.get(edge[0])



        embedding2 = embeddings_dict.get(edge[1])



        embList = []

        for i in range(len(embedding1)):

            emb_feature =(embedding1[i]+embedding2[i])/2



            embList.append(emb_feature)

        embeddingEdgesList.append(embList)

        edgeExistList.append(1)

    alreadyaddedembeddings = len(embeddingEdgesList)

    addedembeddings = 0

    for nodeNum in range(len(nodesList)):

        node1 = nodesList[nodeNum]

        node2Num = random.randint(0, len(nodesList)-1)

        while (nodeNum == node2Num):

            node2Num = random.randint(0, len(nodesList)-1)

        node2 = nodesList[node2Num]

        edge = (node1,node2)



        reverse_edge = (node2,node1)



        while (edge in edgeList or reverse_edge in edgeList or edge in deletedEdgesList or reverse_edge in deletedEdgesList):

            node2Num = random.randint(0, len(nodesList)-1)

            while (nodeNum == node2Num):

                node2Num = random.randint(0, len(nodesList)-1)

            node2 = nodesList[node2Num]

            edge = (node1,node2)

            reverse_edge = (node2,node1)

        embedding1 = embeddings_dict.get(edge[0])

        embedding2 = embeddings_dict.get(edge[1])

        embList = []

        for i in range(len(embedding1)):



            emb_feature =(embedding1[i]+embedding2[i])/2



            embList.append(emb_feature)

        embeddingEdgesList.append(embList)

        addedembeddings+=1





        edgeExistList.append(0)

        if(addedembeddings>= alreadyaddedembeddings):

            break

    return list(embeddingEdgesList), list(edgeExistList)



def createTestSetAverage(deletedEdgesList,embeddings_dict,nodesList,edgeList,X_train):

    X_test = []

    y_test = []



    for edge in deletedEdgesList:



        embedding1 = embeddings_dict.get(edge[0])

        embedding2 = embeddings_dict.get(edge[1])



        embList = []

        for i in range(len(embedding1)):



            emb_feature =(embedding1[i]+embedding2[i])/2



            embList.append(emb_feature)

        X_test.append(embList)

        y_test.append(1)

    alreadyaddedembeddings = len(X_test)

    addedembeddings = 0

    for nodeNum in range(len(nodesList)):

        node1 = nodesList[nodeNum]

        node2Num = random.randint(0, len(nodesList)-1)

        while (nodeNum == node2Num):

            node2Num = random.randint(0, len(nodesList)-1)

        node2 = nodesList[node2Num]

        edge = (node1,node2)

        reverse_edge = (node2,node1)

        while (edge in edgeList or reverse_edge in edgeList or edge in deletedEdgesList or reverse_edge in deletedEdgesList or edge in X_train or reverse_edge in X_train):

            node2Num = random.randint(0, len(nodesList)-1)

            while (nodeNum == node2Num):

                node2Num = random.randint(0, len(nodesList)-1)

            node2 = nodesList[node2Num]

            edge = (node1,node2)

            reverse_edge = (node2,node1)

        embedding1 = embeddings_dict.get(edge[0])

        embedding2 = embeddings_dict.get(edge[1])


        embList = []

        for i in range(len(embedding1)):



            emb_feature =(embedding1[i]+embedding2[i])/2



            embList.append(emb_feature)

        X_test.append(embList)

        addedembeddings+=1





        y_test.append(0)

        if(addedembeddings>= alreadyaddedembeddings):

            break

    return list(X_test), list(y_test)



def createTrainSetHadamard(edgesList,nodesList,embeddings_dict,deletedEdgesList):

    embeddingEdgesList = []

    edgeExistList = []

    edgeList = []

    for edge in edgesList:

        edgeList.append(edge)

        embedding1 = embeddings_dict.get(edge[0])



        embedding2 = embeddings_dict.get(edge[1])

        embList = []

        for i in range(len(embedding1)):



            emb_feature =(embedding1[i]*embedding2[i])



            embList.append(emb_feature)

        embeddingEdgesList.append(embList)

        edgeExistList.append(1)

    alreadyaddedembeddings = len(embeddingEdgesList)

    addedembeddings = 0

    for nodeNum in range(len(nodesList)):

        node1 = nodesList[nodeNum]

        node2Num = random.randint(0, len(nodesList)-1)

        while (nodeNum == node2Num):

            node2Num = random.randint(0, len(nodesList)-1)

        node2 = nodesList[node2Num]

        edge = (node1,node2)

        reverse_edge = (node2,node1)

        while (edge in edgeList or reverse_edge in edgeList or edge in deletedEdgesList or reverse_edge in deletedEdgesList):

            node2Num = random.randint(0, len(nodesList)-1)

            while (nodeNum == node2Num):

                node2Num = random.randint(0, len(nodesList)-1)

            node2 = nodesList[node2Num]

            edge = (node1,node2)

            reverse_edge = (node2,node1)

        embedding1 = embeddings_dict.get(edge[0])

        embedding2 = embeddings_dict.get(edge[1])

        embList = []

        for i in range(len(embedding1)):



            emb_feature =(embedding1[i]*embedding2[i])



            embList.append(emb_feature)

        embeddingEdgesList.append(embList)

        addedembeddings+=1





        edgeExistList.append(0)

        if(addedembeddings>= alreadyaddedembeddings):

            break

    return list(embeddingEdgesList),list(edgeExistList)



def createTestSetHadamard(deletedEdgesList,embeddings_dict,nodesList,edgeList,X_train):

    X_test = []

    y_test = []



    for edge in deletedEdgesList:

        embedding1 = embeddings_dict.get(edge[0])

        embedding2 = embeddings_dict.get(edge[1])

        embList = []

        for i in range(len(embedding1)):



            emb_feature =(embedding1[i]*embedding2[i])



            embList.append(emb_feature)

        X_test.append(embList)

        y_test.append(1)

    alreadyaddedembeddings = len(X_test)

    addedembeddings = 0

    for nodeNum in range(len(nodesList)):

        node1 = nodesList[nodeNum]

        node2Num = random.randint(0, len(nodesList)-1)

        while (nodeNum == node2Num):

            node2Num = random.randint(0, len(nodesList)-1)

        node2 = nodesList[node2Num]

        edge = (node1,node2)

        reverse_edge = (node2,node1)

        while (edge in edgeList or reverse_edge in edgeList or edge in deletedEdgesList or reverse_edge in deletedEdgesList or edge in X_train or reverse_edge in X_train):

            node2Num = random.randint(0, len(nodesList)-1)

            while (nodeNum == node2Num):

                node2Num = random.randint(0, len(nodesList)-1)

            node2 = nodesList[node2Num]

            edge = (node1,node2)

            reverse_edge = (node2,node1)

        embedding1 = embeddings_dict.get(edge[0])

        embedding2 = embeddings_dict.get(edge[1])

        embList = []

        for i in range(len(embedding1)):



            emb_feature =(embedding1[i]*embedding2[i])



            embList.append(emb_feature)

        X_test.append(embList)

        addedembeddings+=1





        y_test.append(0)

        if(addedembeddings>= alreadyaddedembeddings):

            break

    return list(X_test), list(y_test)





def createTrainSetWeightedL1(edgesList,nodesList,embeddings_dict,deletedEdgesList):

    embeddingEdgesList = []

    edgeExistList = []

    edgeList = []

    for edge in edgesList:

        edgeList.append(edge)

        embedding1 = embeddings_dict.get(edge[0])



        embedding2 = embeddings_dict.get(edge[1])

        embList = []

        for i in range(len(embedding1)):



            emb_feature =abs(embedding1[i]-embedding2[i])



            embList.append(emb_feature)

        embeddingEdgesList.append(embList)

        edgeExistList.append(1)

    alreadyaddedembeddings = len(embeddingEdgesList)

    addedembeddings = 0

    for nodeNum in range(len(nodesList)):

        node1 = nodesList[nodeNum]

        node2Num = random.randint(0, len(nodesList)-1)

        while (nodeNum == node2Num):

            node2Num = random.randint(0, len(nodesList)-1)

        node2 = nodesList[node2Num]

        edge = (node1,node2)

        reverse_edge = (node2,node1)

        while (edge in edgeList or reverse_edge in edgeList or edge in deletedEdgesList or reverse_edge in deletedEdgesList):

            node2Num = random.randint(0, len(nodesList)-1)

            while (nodeNum == node2Num):

                node2Num = random.randint(0, len(nodesList)-1)

            node2 = nodesList[node2Num]

            edge = (node1,node2)

            reverse_edge = (node2,node1)

        embedding1 = embeddings_dict.get(edge[0])

        embedding2 = embeddings_dict.get(edge[1])

        embList = []

        for i in range(len(embedding1)):



            emb_feature =abs(embedding1[i]-embedding2[i])



            embList.append(emb_feature)

        embeddingEdgesList.append(embList)

        addedembeddings+=1





        edgeExistList.append(0)

        if(addedembeddings>= alreadyaddedembeddings):

            break

    return list(embeddingEdgesList),list(edgeExistList)



def createTestSetWeightedL1(deletedEdgesList,embeddings_dict,nodesList,edgeList,X_train):

    X_test = []

    y_test = []



    for edge in deletedEdgesList:

        embedding1 = embeddings_dict.get(edge[0])

        embedding2 = embeddings_dict.get(edge[1])

        embList = []

        for i in range(len(embedding1)):



            emb_feature =abs(embedding1[i]-embedding2[i])



            embList.append(emb_feature)

        X_test.append(embList)

        y_test.append(1)

    alreadyaddedembeddings = len(X_test)

    addedembeddings = 0

    for nodeNum in range(len(nodesList)):

        node1 = nodesList[nodeNum]

        node2Num = random.randint(0, len(nodesList)-1)

        while (nodeNum == node2Num):

            node2Num = random.randint(0, len(nodesList)-1)

        node2 = nodesList[node2Num]

        edge = (node1,node2)

        reverse_edge = (node2,node1)

        while (edge in edgeList or reverse_edge in edgeList or edge in deletedEdgesList or reverse_edge in deletedEdgesList or edge in X_train or reverse_edge in X_train):

            node2Num = random.randint(0, len(nodesList)-1)

            while (nodeNum == node2Num):

                node2Num = random.randint(0, len(nodesList)-1)

            node2 = nodesList[node2Num]

            edge = (node1,node2)

            reverse_edge = (node2,node1)

        embedding1 = embeddings_dict.get(edge[0])

        embedding2 = embeddings_dict.get(edge[1])

        embList = []

        for i in range(len(embedding1)):



            emb_feature =abs(embedding1[i]-embedding2[i])



            embList.append(emb_feature)

        X_test.append(embList)

        addedembeddings+=1





        y_test.append(0)

        if(addedembeddings>= alreadyaddedembeddings):

            break

    return list(X_test), list(y_test)





def createTrainSetWeightedL2(edgesList,nodesList,embeddings_dict,deletedEdgesList):

    embeddingEdgesList = []

    edgeExistList = []

    edgeList = []

    for edge in edgesList:

        edgeList.append(edge)

        embedding1 = embeddings_dict.get(edge[0])



        embedding2 = embeddings_dict.get(edge[1])

        embList = []

        for i in range(len(embedding1)):



            emb_feature =(abs(embedding1[i]-embedding2[i]))**2



            embList.append(emb_feature)

        embeddingEdgesList.append(embList)

        edgeExistList.append(1)

    alreadyaddedembeddings = len(embeddingEdgesList)

    addedembeddings = 0

    for nodeNum in range(len(nodesList)):

        node1 = nodesList[nodeNum]

        node2Num = random.randint(0, len(nodesList)-1)

        while (nodeNum == node2Num):

            node2Num = random.randint(0, len(nodesList)-1)

        node2 = nodesList[node2Num]

        edge = (node1,node2)

        reverse_edge = (node2,node1)

        while (edge in edgeList or reverse_edge in edgeList or edge in deletedEdgesList or reverse_edge in deletedEdgesList):

            node2Num = random.randint(0, len(nodesList)-1)

            while (nodeNum == node2Num):

                node2Num = random.randint(0, len(nodesList)-1)

            node2 = nodesList[node2Num]

            edge = (node1,node2)

            reverse_edge = (node2,node1)

        embedding1 = embeddings_dict.get(edge[0])

        embedding2 = embeddings_dict.get(edge[1])

        embList = []

        for i in range(len(embedding1)):



            emb_feature =(abs(embedding1[i]-embedding2[i]))**2



            embList.append(emb_feature)

        embeddingEdgesList.append(embList)

        addedembeddings+=1





        edgeExistList.append(0)

        if(addedembeddings>= alreadyaddedembeddings):

            break

    return list(embeddingEdgesList),list(edgeExistList)



def createTestSetWeightedL2(deletedEdgesList,embeddings_dict,nodesList,edgeList, X_train):

    X_test = []

    y_test = []



    for edge in deletedEdgesList:

        embedding1 = embeddings_dict.get(edge[0])

        embedding2 = embeddings_dict.get(edge[1])

        embList = []

        for i in range(len(embedding1)):



            emb_feature =(abs(embedding1[i]-embedding2[i]))**2



            embList.append(emb_feature)

        X_test.append(embList)

        y_test.append(1)

    alreadyaddedembeddings = len(X_test)

    addedembeddings = 0

    for nodeNum in range(len(nodesList)):

        node1 = nodesList[nodeNum]

        node2Num = random.randint(0, len(nodesList)-1)

        while (nodeNum == node2Num):

            node2Num = random.randint(0, len(nodesList)-1)

        node2 = nodesList[node2Num]

        edge = (node1,node2)

        reverse_edge = (node2,node1)

        while (edge in edgeList or reverse_edge in edgeList or edge in deletedEdgesList or reverse_edge in deletedEdgesList or edge in X_train or reverse_edge in X_train):

            node2Num = random.randint(0, len(nodesList)-1)

            while (nodeNum == node2Num):

                node2Num = random.randint(0, len(nodesList)-1)

            node2 = nodesList[node2Num]

            edge = (node1,node2)

            reverse_edge = (node2,node1)

        embedding1 = embeddings_dict.get(edge[0])

        embedding2 = embeddings_dict.get(edge[1])

        embList = []

        for i in range(len(embedding1)):



            emb_feature =(abs(embedding1[i]-embedding2[i]))**2



            embList.append(emb_feature)

        X_test.append(embList)

        addedembeddings+=1





        y_test.append(0)

        if(addedembeddings>= alreadyaddedembeddings):

            break

    return list(X_test), list(y_test)





def createEmbeddingEdgeAverage(edge,embeddings_dict):

    embedding1 = embeddings_dict.get(edge[0])

    embedding2 = embeddings_dict.get(edge[1])

    embList = []

    for i in range(len(embedding1)):

        emb_feature =(embedding1[i]+embedding2[i])/2

        embList.append(emb_feature)

    return embList



def createEmbeddingEdgeHadamard(edge,embeddings_dict):

    embedding1 = embeddings_dict.get(edge[0])

    embedding2 = embeddings_dict.get(edge[1])

    embList = []

    for i in range(len(embedding1)):

        emb_feature =(embedding1[i]*embedding2[i])

        embList.append(emb_feature)

    return embList





def createEmbeddingEdgeWeightedL1(edge,embeddings_dict):

    embedding1 = embeddings_dict.get(edge[0])

    embedding2 = embeddings_dict.get(edge[1])

    embList = []

    for i in range(len(embedding1)):

        emb_feature = abs(embedding1[i]-embedding2[i])

        embList.append(emb_feature)

    return embList



def createEmbeddingEdgeWeightedL2(edge,embeddings_dict):

    embedding1 = embeddings_dict.get(edge[0])

    embedding2 = embeddings_dict.get(edge[1])

    embList = []

    for i in range(len(embedding1)):

        emb_feature =(abs(embedding1[i]-embedding2[i]))**2

        embList.append(emb_feature)

    return embList
