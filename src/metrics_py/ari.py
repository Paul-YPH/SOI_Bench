###############################################
################## Embedding ##################
###############################################

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari_score

def compute_ari(adata,
                label_key = 'Ground Truth',
                cluster_key = 'benchmark_cluster'):

    # Compute ARI for each cluster key (defferent clustering methods)
    true_labels = adata.obs[label_key]
    predicted_labels = adata.obs[cluster_key]
    ari = ari_score(true_labels, predicted_labels)
    
    df = pd.DataFrame({'metric': 'ari', 'value': [ari], 'group': [cluster_key]})
    return df
