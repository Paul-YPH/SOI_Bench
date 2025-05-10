import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc

from paste2 import PASTE2, projection
from paste2.model_selection import select_overlap_fraction

#################### Run PASTE2 ####################

def run_PASTE2(ann_list, sample_list, overlap_list):
    pi_list = []
    matching_cell_ids_list = []
    
    if overlap_list is None:
        overlap_list = [0.99] * (len(sample_list) - 1)
    else:
        overlap_list = []
        for i in range(len(sample_list)-1):
            j = i+1
            overlap_list.append(ann_list[i].shape[0]/ann_list[j].shape[0])
        
    for i in range(len(sample_list)-1):
        j = i+1
        adata1 = ann_list[i]
        adata2 = ann_list[j]    

        # l = adata1.copy()
        # siml = adata2.copy()
        #overlap_frac = select_overlap_fraction(l, siml, alpha=0.1)
        pi12 = PASTE2.partial_pairwise_align(adata1, adata2, s=overlap_list[i],dissimilarity = 'pca')

        result = pd.DataFrame(pi12)
        cell_ids_1 = adata1.obs.index.to_numpy()
        cell_ids_2 = adata2.obs.index.to_numpy() 

        result.index = cell_ids_1
        result.columns = cell_ids_2

        matching_index = np.argmax(result.to_numpy(), axis=0)
        matching_cell_ids = pd.DataFrame({
            'cell_id_1': cell_ids_1[matching_index],
            'cell_id_2': cell_ids_2   
        })
        pi_list.append(result)
        matching_cell_ids_list.append(matching_cell_ids)

    pi_array_list = [pi.to_numpy() for pi in pi_list]
    new_slices = projection.partial_stack_slices_pairwise(ann_list, pi_array_list)
    adata = new_slices[0]
    for i in range(1, len(new_slices)):
        adata = adata.concatenate(new_slices[i], index_unique=None, batch_key=None)
    adata.obsm['spatial_aligned'] = adata.obsm['spatial']
    
    if 'high_quality_transfer' in adata.obs.columns:
        adata.obs['high_quality_transfer'] = adata.obs['high_quality_transfer'].astype(str)

    return adata, pi_list, matching_cell_ids_list
