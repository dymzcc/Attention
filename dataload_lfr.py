import math
import pandas as pd
import networkx as nx
import numpy as np
import dgl
from sklearn.metrics import f1_score
from networkx.algorithms.community import asyn_fluidc


def load_lfr_1():
    a = pd.read_table('/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/one/1/network.txt', sep='\s+',
                      header=None)
    x = a[0].values
    y = a[1].values
    g = nx.Graph()
    for n in range(0, len(x)):
        g.add_node(x[n])
        g.add_edge(x[n], y[n])
    return g

def get_label_lfr_1():
    b = '/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/one/1/community.txt'
    data = []
    with open(b, 'r') as f:
        line = f.readline()
        while line:
            eachline = line.split()
            read_data = [int(x) for x in eachline[0:]]
            data.append(read_data)
            line = f.readline()
    label = np.zeros((1000, 11))
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            label[i][data[i][j]-1] = 1
    return label

#def get_feature_lfr_1():


def threshold(g):
    num_nodes = g.number_of_nodes()
    num_edges = g.number_of_edges()
    e = 2*num_edges/(num_nodes*(num_edges-1))
    thr = np.sqrt(-math.log(1-e))
    return thr

def get_feature_lfr_1():

    a = pd.read_table('/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/one/1/feature.txt', sep='\s+',
                      header=None)
    x = list(a[0].values)
    y = list(a[1].values)
    z = list(zip(x,y))
    z.sort(key=(lambda x:x[0]))

    feature = np.zeros((1000, 11))
    for n in range(0, len(z)):
        feature[z[n][0] - 1, z[n][1] - 1] = 1

    feature = np.hstack((feature,feature,feature,feature,feature,feature,feature,feature))
    return feature

def Neighbor():
    g = load_lfr_1()
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
    g = dgl.from_networkx(load_lfr_1())
    u = g.edges()[0]
    v = g.edges()[1]
    weight = []

    for i in range(g.number_of_edges()):
        ret_1 = list(set(neighbors[u[i]]).intersection(set(neighbors[v[i]])))
        ret_2 = list(set(neighbors[u[i]]).union(set(neighbors[v[i]])))
        weight.append((len(ret_1) + 1) / (len(ret_2) + 1))
    return weight

def load_lfr_2():
    a = pd.read_table('/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/one/2/network.txt', sep='\s+',
                      header=None)
    x = a[0].values
    y = a[1].values
    g = nx.Graph()
    for n in range(0, len(x)):
        g.add_node(x[n])
        g.add_edge(x[n], y[n])
    return g

def get_label_lfr_2():
    b = '/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/one/2/community.txt'
    data = []
    with open(b, 'r') as f:
        line = f.readline()
        while line:
            eachline = line.split()
            read_data = [int(x) for x in eachline[0:]]
            data.append(read_data)
            line = f.readline()
    label = np.zeros((1000, 11))
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            label[i][data[i][j]-1] = 1
    return label

def get_feature_lfr_2():

    a = pd.read_table('/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/one/2/feature.txt', sep='\s+',
                      header=None)
    x = list(a[0].values)
    y = list(a[1].values)
    z = list(zip(x,y))
    z.sort(key=(lambda x:x[0]))

    feature = np.zeros((1000, 11))
    for n in range(0, len(z)):
        feature[z[n][0] - 1, z[n][1] - 1] = 1

    feature = np.hstack((feature,feature,feature,feature,feature,feature,feature,feature))
    return feature

def load_lfr_3():
    a = pd.read_table('/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/one/3/network.txt', sep='\s+',
                      header=None)
    x = a[0].values
    y = a[1].values
    g = nx.Graph()
    for n in range(0, len(x)):
        g.add_node(x[n])
        g.add_edge(x[n], y[n])
    return g

def get_label_lfr_3():
    b = '/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/one/3/community.txt'
    data = []
    with open(b, 'r') as f:
        line = f.readline()
        while line:
            eachline = line.split()
            read_data = [int(x) for x in eachline[0:]]
            data.append(read_data)
            line = f.readline()
    label = np.zeros((1000, 11))
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            label[i][data[i][j]-1] = 1
    return label

def get_feature_lfr_3():

    a = pd.read_table('/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/one/3/feature.txt', sep='\s+',
                      header=None)
    x = list(a[0].values)
    y = list(a[1].values)
    z = list(zip(x,y))
    z.sort(key=(lambda x:x[0]))

    feature = np.zeros((1000, 11))
    for n in range(0, len(z)):
        feature[z[n][0] - 1, z[n][1] - 1] = 1

    feature = np.hstack((feature,feature,feature,feature,feature,feature,feature,feature))
    return feature