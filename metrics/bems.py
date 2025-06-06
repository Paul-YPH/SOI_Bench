###############################################
################## Embedding ##################
###############################################

# The code is modified from SCALEX and https://github.com/caokai1073/uniPort/blob/main/uniport/metrics.py

from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy
import pandas as pd
from .utils import validate_adata, check_adata

def compute_bems(adata, 
                 use_rep,
                 batch_key, 
                 n_neighbors=100, 
                 n_pools=100, 
                 n_samples_per_pool=100):
    
    data = np.array(adata.obsm[use_rep])
    batches = np.array(adata.obs[batch_key])
    
#     print("Start calculating Entropy mixing score")
    def entropy(batches):
        p = np.zeros(N_batches)
        adapt_p = np.zeros(N_batches)
        a = 0
        for i in range(N_batches):
            p[i] = np.mean(batches == batches_[i])
            a = a + p[i]/P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i]/P[i])/a
            entropy = entropy - adapt_p[i]*np.log(adapt_p[i]+10**-8)
        return entropy

    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    score = 0
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    for i in range(N_batches):
            P[i] = np.mean(batches == batches_[i])
    for t in range(n_pools):
        indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
                                                 [kmatrix[indices].nonzero()[0] == i]])
                          for i in range(n_samples_per_pool)])
    Score = score / float(n_pools)
    value = Score / float(np.log2(N_batches))
    df = pd.DataFrame({'metric': 'bems', 'value': [value], 'group': [batch_key]})
    return df