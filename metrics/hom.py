###############################################
################## Embedding ##################
###############################################


import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_score as hom_score

def compute_hom(adata,
                label_key,
                cluster_key):

    if cluster_key not in adata.obs.columns:
        raise ValueError(f"The cluster key {cluster_key} is not in `adata.obs`.")

    # Compute HOM for each cluster key
    true_labels = adata.obs[label_key]
    results = []
    
    predicted_labels = adata.obs[cluster_key]
    value = hom_score(true_labels, predicted_labels)
    
    results.append({'metric': 'hom', 'value': value, 'group': cluster_key})
    df = pd.DataFrame(results)
    return df
