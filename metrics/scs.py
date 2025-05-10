#########################################################
################## Alignment/Embedding ##################
#########################################################

import math
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import networkx as nx
from scipy.spatial import distance_matrix
import random
from scipy.spatial import KDTree

# https://github.com/raphael-group/paste_reproducibility/blob/6c5a6ef456cc49a41135863f16e6c26b5585a6fd/notebooks/visium-scc-analysis.ipynb#L86

def compute_scs(adata, 
                label_key = None,
                cluster_key = None,
                batch_key = None,
                spatial_key = 'spatial'):
    
    if cluster_key is not None:
        key = cluster_key   
        results = []
        for batch in adata.obs[batch_key].unique():
            tmp = adata[adata.obs[batch_key] == batch]
            scs_value = spatial_coherence_score_optimized(tmp, label_key = key, spatial_key=spatial_key)
            results.append({'metric': 'scs', 'value': scs_value, 'group': f'{batch}'})    
        df = pd.DataFrame(results)
    elif label_key is not None:
        key = label_key
        scs_value = spatial_coherence_score_optimized(adata, label_key = key, spatial_key=spatial_key)
        df = pd.DataFrame({'metric': 'scs', 'value': [scs_value], 'group': [key]})
    else:
        raise ValueError("Either `cluster_key` or `label_key` must be provided.")
    return df

# def create_graph(adata, degree = 4,spatial_key = 'spatial'):
#         """
#         Converts spatial coordinates into graph using networkx library.
        
#         param: adata - ST Slice 
#         param: degree - number of edges per vertex

#         return: 1) G - networkx graph
#                 2) node_dict - dictionary mapping nodes to spots
#         """
#         D = distance_matrix(adata.obsm[spatial_key], adata.obsm[spatial_key])
#         # Get column indexes of the degree+1 lowest values per row
#         idx = np.argsort(D, 1)[:, 0:degree+1]
#         # Remove first column since it results in self loops
#         idx = idx[:, 1:]

#         G = nx.Graph()
#         for r in range(len(idx)):
#             for c in idx[r]:
#                 G.add_edge(r, c)

#         node_dict = dict(zip(range(adata.shape[0]), adata.obs.index))
#         return G, node_dict
def create_graph(adata, degree=4, spatial_key='spatial'):
    tree = KDTree(adata.obsm[spatial_key])  # 构建 KDTree
    _, idx = tree.query(adata.obsm[spatial_key], k=degree+1) 
    idx = idx[:, 1:]

    G = nx.Graph()
    for r in range(len(idx)):
        for c in idx[r]:
            G.add_edge(r, c)

    node_dict = dict(zip(range(adata.shape[0]), adata.obs.index))
    return G, node_dict
    
def generate_graph_from_labels(adata, label_key=None, spatial_key = 'spatial'):
    """
    Creates and returns the graph and dictionary {node: cluster_label} for specified layer
    """
    
    g, node_to_spot = create_graph(adata, spatial_key = spatial_key)
    
    spot_to_cluster = adata.obs[label_key]

    # remove any nodes that are not mapped to a cluster
    removed_nodes = []
    for node in node_to_spot.keys():
        if (node_to_spot[node] not in spot_to_cluster.keys()):
            removed_nodes.append(node)

    for node in removed_nodes:
        del node_to_spot[node]
        g.remove_node(node)
        
    labels = dict(zip(g.nodes(), [spot_to_cluster[node_to_spot[node]] for node in g.nodes()]))
    return g, labels

# def spatial_coherence_score(adata, label_key=None, spatial_key = 'spatial'):
    
#     g, l = generate_graph_from_labels(adata, label_key, spatial_key)
    
#     true_entropy = spatial_entropy(g, l)
#     entropies = []
#     for i in range(1000):
#         new_l = list(l.values())
#         random.shuffle(new_l)
#         labels = dict(zip(l.keys(), new_l))
#         entropies.append(spatial_entropy(g, labels))
        
#     return (true_entropy - np.mean(entropies))/np.std(entropies)

def spatial_entropy(g, labels):
    """
    Calculates spatial entropy of graph  
    """
    # construct contiguity matrix C which counts pairs of cluster edges
    cluster_names = np.unique(list(labels.values()))
    C = pd.DataFrame(0,index=cluster_names, columns=cluster_names)

    for e in g.edges():
        C.loc[labels[e[0]], labels[e[1]]] += 1

    # calculate entropy from C
    C_sum = C.values.sum()
    H = 0
    for i in range(len(cluster_names)):
        for j in range(i, len(cluster_names)):
            if (i == j):
                z = C[cluster_names[i]][cluster_names[j]]
            else:
                z = C[cluster_names[i]][cluster_names[j]] + C[cluster_names[j]][cluster_names[i]]
            if z != 0:
                H += -(z/C_sum)*math.log(z/C_sum)
    return H

def permutation_entropy(g, labels, keys):
    new_l = list(labels.values())
    random.shuffle(new_l)
    perm_labels = dict(zip(keys, new_l))
    return spatial_entropy(g, perm_labels)

def spatial_coherence_score_optimized(adata, label_key=None, spatial_key='spatial', n_permutations=1000):
    g, l = generate_graph_from_labels(adata, label_key, spatial_key)  
    true_entropy = spatial_entropy(g, l)
    keys = list(l.keys())
    entropies = Parallel(n_jobs=-1)(delayed(permutation_entropy)(g, l, keys) for _ in range(n_permutations))
    return (true_entropy - np.mean(entropies)) / np.std(entropies)