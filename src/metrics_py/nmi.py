###############################################
################## Embedding ##################
###############################################


import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi_score

def compute_nmi(adata,
                label_key = 'Ground Truth',
                cluster_key = 'benchmark_cluster'):

    # Compute NMI for each cluster key
    true_labels = adata.obs[label_key]
    results = []

    predicted_labels = adata.obs[cluster_key]
    value = nmi_score(true_labels, predicted_labels)
    results.append({'metric': 'nmi', 'value': value, 'group': cluster_key})
    df = pd.DataFrame(results)
    return df