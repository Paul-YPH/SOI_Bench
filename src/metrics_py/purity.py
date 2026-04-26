###############################################
################## Embedding ##################
###############################################

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import contingency_matrix

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm) 

def compute_purity(adata,
                label_key = 'Ground Truth',
                cluster_key = 'benchmark_cluster'):
    true_labels = np.array(adata.obs[label_key])
    predicted_labels = np.array(adata.obs[cluster_key])
    purity = purity_score(true_labels, predicted_labels)
    df = pd.DataFrame({'metric': 'purity', 'value': [purity], 'group': [cluster_key]})
    return df
