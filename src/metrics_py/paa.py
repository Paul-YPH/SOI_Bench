###############################################
################## Alignment ##################
###############################################

import numpy as np
import pandas as pd
import scanpy as sc

def compute_paa(adata, 
                batch_key='batch',
                label_key='Ground Truth'):
    def ensure_list(data):
        if isinstance(data, (dict, pd.Series)):
            return [data[str(i)] for i in range(len(data))]
        return data
    matching_list = ensure_list(adata.uns['matching_cell_ids_list'])
    matching_list = [pd.DataFrame(item, columns=['id_1', 'id_2']) for item in matching_list]
        
    batch_list = adata.uns['config']['sample_list']

    results = []
    for i in range(len(batch_list) - 1):
        batch1, batch2 = batch_list[i], batch_list[i + 1]
        df = matching_list[i]

        adata1 = adata[adata.obs[batch_key] == batch1]
        adata2 = adata[adata.obs[batch_key] == batch2]
        
        meta1 = adata1.obs[[label_key]].copy()
        meta1['id_1'] = meta1.index
        meta2 = adata2.obs[[label_key]].copy()
        meta2['id_2'] = meta2.index
        
        all_categories = pd.unique(meta1[label_key].tolist() + meta2[label_key].tolist()).tolist()
        meta1[label_key] = pd.Categorical(meta1[label_key], categories=all_categories)
        meta2[label_key] = pd.Categorical(meta2[label_key], categories=all_categories)

        df1 = df.merge(meta1, left_on='id_1', right_index=True, how='inner')
        df2 = df.merge(meta2, left_on='id_2', right_index=True, how='inner')

        common_index = df1.index.intersection(df2.index)
        if len(common_index) == 0:
            print(f"Warning: No common matches for {batch1}_{batch2}, accuracy set to 0")
            results.append({'metric': 'paa', 'value': 0, 'group': f'{batch1}_{batch2}'})
            continue

        df1 = df1.loc[common_index]
        df2 = df2.loc[common_index]

        correct_matches = (df1[label_key] == df2[label_key]).sum()
        print(f"Correct Matches for {batch1}_{batch2}: {correct_matches}")

        total_labels = len(common_index)
        accuracy = correct_matches / total_labels if total_labels > 0 else 0

        results.append({'metric': 'paa', 'value': accuracy, 'group': f'{batch1}_{batch2}'})

    df = pd.DataFrame(results)
    return df