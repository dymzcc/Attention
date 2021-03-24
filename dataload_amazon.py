import math
import pandas as pd
import networkx as nx
import numpy as np
import dgl

def load_amazon():
    a = pd.read_table('/Users/fartinhands/Desktop/GDC/Dataset/dataset/real_dataset/amazon/com-amazon_agm.txt', sep='\s+',
                      header=None)
    x = a[0].values
    y = a[1].values
    g = nx.Graph()
    for n in range(0, len(x)):
        g.add_node(x[n]+1)
        g.add_edge(x[n]+1, y[n]+1)
    return g

def get_label_amazon():
    b = '/Users/fartinhands/Desktop/GDC/Dataset/dataset/real_dataset/amazon/com-amazon.sampled.cmty.txt'
    data = []
    with open(b, 'r') as f:
        line = f.readline()
        while line:
            eachline = line.split()
            read_data = [int(x) for x in eachline[0:]]
            data.append(read_data)
            line = f.readline()
    label = np.zeros((3225, 100))
    for i in range(len(data)):
        for j in range(len(data[i])):
            label[data[i][j]][i] = 1
    return label

def get_feature_amazon():
    c = '/Users/fartinhands/Desktop/GDC/Dataset/dataset/real_dataset/amazon/com-amazon_pre_train.txt'
    data = []
    with open(c, 'r') as f:
        line = f.readline()
        while line:
            eachline = line.split()
            read_data = [float(x) for x in eachline[0:]]
            data.append(read_data)
            line = f.readline()
    data.sort(key=(lambda data: data[0]))
    feature = np.array(data)
    feature = np.delete(feature, 0, axis=1)
    for n in range(len(feature)):
        for i in range(len(feature[n])):
            if feature[n][i] > 0: feature[n][i] = 1
    #feature = np.hstack((feature, feature, feature))
    return feature

def threshold(g):
    num_nodes = g.number_of_nodes()
    num_edges = g.number_of_edges()
    e = 2*num_edges/(num_nodes*(num_edges-1))
    thr = np.sqrt(-math.log(1-e))
    return thr

def Neighbor():
    g = load_amazon()
    neighbors = []
    for node in range(1,g.number_of_nodes()+1):
        list = []
        for i in g.neighbors(node):
            list.append(i)
        neighbors.append(list)
    return neighbors

def egde_pro():

    neighbors = Neighbor()
    g = dgl.DGLGraph()
    g = dgl.from_networkx(load_amazon())
    u = g.edges()[0]
    v = g.edges()[1]
    weight = []

    for i in range(g.number_of_edges()):
        ret_1 = list(set(neighbors[u[i]]).intersection(set(neighbors[v[i]])))
        ret_2 = list(set(neighbors[u[i]]).union(set(neighbors[v[i]])))
        weight.append((len(ret_1) + 1) / (len(ret_2) + 1))
    return weight
