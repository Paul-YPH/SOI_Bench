###############################################
################## Embedding ##################
###############################################

# The code is modified from https://github.com/theislab/scib/blob/main/scib/metrics/lisi.py

import numpy as np
import pandas as pd
import sklearn.metrics

def compute_silhouette(
    adata,
    use_rep = None,
    batch_key = None,
    label_key = None,
    compute_fn = None,
    **kwargs
):
    
    if compute_fn == 'asw_batch':
        if label_key is None:
            raise ValueError("Cell type information must be provided for batch silhouette score.")
        
        x = np.array(adata.obsm[use_rep])
        y = np.array(adata.obs[batch_key])
    
        scores_per_group = []
        celltype = np.array(adata.obs[label_key])
        
        for label in np.unique(celltype):
            mask = celltype == label
            try:
                scores = sklearn.metrics.silhouette_samples(x[mask], y[mask], **kwargs)
            except ValueError:  # Too few samples
                scores = 0
            scores = (1 - np.abs(scores)).mean()
            scores_per_group.append(scores)
        score = np.mean(scores_per_group)
        
    elif compute_fn == 'asw_celltype':
        
        x = np.array(adata.obsm[use_rep])
        y = np.array(adata.obs[label_key])
        
        score = (sklearn.metrics.silhouette_score(x, y, **kwargs).item() + 1) / 2
    else:
        raise ValueError("Invalid `compute_fn`. Must be 'asw_celltype' or 'asw_batch'.")

    return score

def compute_asw_celltype(
    adata,
    use_rep = 'integrated',
    batch_key = 'batch',
    **kwargs
):  
    value = compute_silhouette(adata, use_rep, batch_key, compute_fn='asw_celltype', **kwargs)
    df = pd.DataFrame({'metric': 'asw_celltype', 'value': [value], 'group': [batch_key]})
    return df

def compute_asw_batch(
    adata,
    use_rep = 'integrated',
    batch_key = 'batch',
    label_key = None,
    **kwargs
):
    value = compute_silhouette(adata, use_rep, batch_key, label_key=label_key, compute_fn='asw_batch', **kwargs)
    df = pd.DataFrame({'metric': 'asw_batch', 'value': [value], 'group': [batch_key]})
    return df

def compute_asw_f1(asw_batch_df, asw_celltype_df):
    asw_batch = asw_batch_df['value'].values
    asw_celltype = asw_celltype_df['value'].values
    # Ensure inputs are within valid range
    if not (0 <= asw_batch <= 1 and 0 <= asw_celltype <= 1):
        raise ValueError("asw_batch and asw_celltype must be in the range [0, 1].")
    
    # Compute the score
    part1 = 2 * asw_batch * asw_celltype
    part2 = asw_batch + asw_celltype
    
    # Avoid division by zero
    if part2 == 0:
        return 0  # Or handle this case differently if required
    
    f1_score = part1 / part2
    df = pd.DataFrame({'metric': 'asw_f1', 'value': f1_score, 'group': ['f1']})
    return df