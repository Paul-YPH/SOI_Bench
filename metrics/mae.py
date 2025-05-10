###############################################
################## Alignment ##################
###############################################

import numpy as np
import pandas as pd

def compute_mae(adata, 
                spatial_aligned_key='spatial_aligned',
                spatial_original_key='spatial_original'
                ):
    mae_value = mae_coordinates(adata.obsm[spatial_aligned_key], adata.obsm[spatial_original_key])
    df = pd.DataFrame({'metric': 'mae', 'value': mae_value, 'group': ['mae']})
    return df

# https://github.com/aristoteleo/spateo-release/blob/a487c87bc0abcaf1c58344635c287140bf8b79e9/spateo/tools/CCI_effects_modeling/regression_utils.py#L856
def mae_coordinates(y_true, y_pred) -> float:
    distances = np.linalg.norm(y_true - y_pred, axis=1)
    return np.mean(distances)