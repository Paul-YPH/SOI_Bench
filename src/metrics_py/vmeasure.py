###############################################
################## Embedding ##################
###############################################

import numpy as np
import pandas as pd
from sklearn.metrics import v_measure_score

def compute_vmeasure(adata,
                    label_key = 'Ground Truth',
                    cluster_key = 'benchmark_cluster'):
    true_labels = np.array(adata.obs[label_key])
    predicted_labels = np.array(adata.obs[cluster_key])
    vmeasure = v_measure_score(true_labels, predicted_labels)
    df = pd.DataFrame({'metric': 'vmeasure', 'value': [vmeasure], 'group': [cluster_key]})
    return df
