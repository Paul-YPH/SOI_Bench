###############################################
################## Embedding ##################
###############################################


import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari_score

def compute_ari(adata,
                label_key,
                cluster_key):

    if cluster_key not in adata.obs.columns:
        raise ValueError(f"The cluster key {cluster_key} is not in `adata.obs`.")

    # Compute ARI for each cluster key (defferent clustering methods)
    true_labels = adata.obs[label_key]
    predicted_labels = adata.obs[cluster_key]
    ari = ari_score(true_labels, predicted_labels)
    
    df = pd.DataFrame({'metric': 'ari', 'value': [ari], 'group': [cluster_key]})
    return df
