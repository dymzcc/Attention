import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from sklearn.model_selection import train_test_split
import networkx as nx
import torch.nn.functional as F
from torch import optim
from nmi import onmi
import tqdm
from dataload_lfr import load_lfr_1,get_feature_lfr_1,get_label_lfr_1,threshold
from dataload_lfr import  load_lfr_2,get_label_lfr_2,get_feature_lfr_2
from dataload_lfr import load_lfr_3,get_label_lfr_3,get_feature_lfr_3
import numpy as np
import scipy.sparse as sp
import torch as th
import dgl


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

def compute_acc(pred, label,g):
    """
    计算准确率
    """
    thr = threshold(g)
    print(thr)
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            if(pred[i][j] > 4*thr):pred[i][j] = 1
            else: pred[i][j] = 0

    pre_com = []
    for j in range(len(pred[0])):
        com = []
        for i in range(len(pred)):
            if (pred[i][j] == 1):
                com.append(i)
        pre_com.append(set(com))

    lab_com = []
    for j in range(len(label[0])):
        com = []
        for i in range(len(label)):
            if (label[i][j] == 1):
                com.append(i)
        lab_com.append(set(com))

    nmi = onmi(pre_com, lab_com)

    emb = pred
    embed_m = sp.csr_matrix(emb.T, dtype=np.uint32)

    label_m = sp.csr_matrix(label.T, dtype=np.uint32)

    n = (label_m.dot(embed_m.T)).toarray().astype(float)  # cg * cd
    p = n / np.array(embed_m.sum(axis=1)).clip(min=1).reshape(-1)
    r = n / np.array(label_m.sum(axis=1)).clip(min=1).reshape(-1, 1)
    f1 = 2 * p * r / (p + r).clip(min=1e-10)
    f1_s1 = f1.max(axis=1).mean()
    f1_s2 = f1.max(axis=0).mean()
    f1_s = (f1_s1 + f1_s2) / 2
    # f1_s = f1_score(label,pred,average="micro")
    return f1_s, nmi

def evaluate(model, feature, subgraph):
    """
    评估模型，调用 model 的 inference 函数
    """
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(feature.float())
        return output
    # model.eval()
    # with th.no_grad():
    #     logits = model(feature)
    #     logits = logits[nodes]
    #     label = label[X_text]
    # #model.train()
    # #return compute_acc(logits,label,g)
    # return logits
    # #return compute_acc(pred[val_mask], labels[val_mask])


if __name__ == '__main__':
    device = th.device('cpu')
    num_epochs = 1000
    num_hidden = 11
    num_layers = 2
    batch_size = 1000
    log_every = 20  # 记录日志的频率
    eval_every = 5
    lr = 0.003
    dropout = 0
    num_heads = 8
    num_out_heads = 1
    m = nn.Sigmoid()
    mm = nn.Softmax(dim=1)


    G = load_lfr_1()
    G.remove_edges_from(nx.selfloop_edges(G))
    g = dgl.DGLGraph()
    g = dgl.from_networkx(G)
    g.add_edges(g.nodes(), g.nodes())
    n_edges = g.number_of_edges()
    heads = ([num_heads] * num_layers) + [num_out_heads]

    in_drop = 0.5
    attn_drop = 0
    negative_slope = 0.2
    residual = False
    n_classes = 11
    weight_decay = 1e-5

    feature = th.Tensor(get_feature_lfr_1())
    in_feats = feature.shape[1]
    label = th.Tensor(get_label_lfr_1())

    X_train, X_text = train_test_split(g.nodes(), test_size=0.75)

    model = GAT(g,
                num_layers,
                in_feats,
                num_hidden,
                n_classes,
                heads,
                F.elu,
                in_drop,
                attn_drop,
                negative_slope,
                residual)

    loss_fcn = nn.BCELoss()
    #loss_fcn = loss_fcn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        logits = m(model(feature))
        loss = loss_fcn(logits[X_train], label[X_train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pred_1 = evaluate(model, feature, g)
    f1_1,nmi_1 = compute_acc(pred_1,label,g)

    g2 = dgl.DGLGraph()
    g2 = dgl.from_networkx(load_lfr_2())
    label_2 = th.Tensor(get_label_lfr_2())
    feature_2 = th.Tensor(get_feature_lfr_2())
    pred_2 = evaluate(model, feature_2,g2)
    f1_2, nmi_2 = compute_acc(pred_2, label_2, g)

    g3 = dgl.DGLGraph()
    g3 = dgl.from_networkx(load_lfr_3())
    label_3 = th.Tensor(get_label_lfr_3())
    feature_3 = th.Tensor(get_feature_lfr_3())
    pred_3 = evaluate(model, feature_3,g3)
    f1_3, nmi_3 = compute_acc(pred_3, label_3, g)

    print(f1_1)
    print(f1_2)
    print(f1_3)
    print('************')
    print(nmi_1)
    print(nmi_2)
    print(nmi_3)


