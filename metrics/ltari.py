###############################################
################## Alignment ##################
###############################################

import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari_score

def compute_ltari(matching_list, 
                  adata, 
                  batch_key='batch',
                  label_key='Ground Truth'):
    batch_list = adata.obs[batch_key].unique()
    results = []
    for i in range(len(matching_list)):
        df = matching_list[i]
        adata1 = adata[adata.obs[batch_key] == batch_list[i]]
        adata2 = adata[adata.obs[batch_key] == batch_list[i+1]]
        # Extract metadata
        meta1 = adata1.obs[[label_key]].copy()
        meta1['cell_id_1'] = meta1.index
        meta2 = adata2.obs[[label_key]].copy()
        meta2['cell_id_2'] = meta2.index

        # Merge metadata with df
        df1 = df.merge(meta1, left_on='cell_id_1', right_index=True, how='inner')
        df2 = df.merge(meta2, left_on='cell_id_2', right_index=True, how='inner')
        
        # Align indices for comparison
        common_index = df1.index.intersection(df2.index)
        df1 = df1.loc[common_index]
        df2 = df2.loc[common_index]
        
        value = ari_score(df1[label_key].astype(str), df2[label_key].astype(str))
        results.append({'metric': 'ltari', 'value': value, 'group': f'{batch_list[i]}_{batch_list[i+1]}'})
    df = pd.DataFrame(results)
    return df