import warnings
warnings.filterwarnings("ignore")

import moscot
from moscot.problems.space import AlignmentProblem
import scanpy as sc
import numpy as np
import pandas as pd
from utils import get_ann_list, create_lightweight_adata,set_seed
import gc

def run_moscot_NR(adata, **args_dict):
    seed = args_dict['seed']
    set_seed(seed)
    
    sample_list = adata.uns['config']['sample_list']
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()

    adata = sc.concat(ann_list, join='inner', index_unique=None)
    for ann in ann_list:
        ann.X = None
        ann.layers.clear()
        ann.raw = None         
    gc.collect()
    
    adata.obs['batch'] = pd.Categorical(adata.obs['batch'], categories=sample_list, ordered=True)

    ap = AlignmentProblem(adata=adata)
    ap = ap.prepare(batch_key="batch", policy="sequential")
    ap = ap.solve(alpha=0.5) ###Spateo
    ap.align(reference=ann_list[0].obs['batch'].unique()[0], key_added="spatial_aligned")
    
    solutions = ap.solutions
    pis = list(solutions.values())
    for i in range(len(pis)):
        pis[i] = list(solutions.values())[i].transport_matrix
        
    pi_list = []
    pi_index_list = []
    pi_column_list = []
    matching_cell_ids_list = []

    for i in range(len(pis)):
        
        index1 = ann_list[i].obs['batch'].unique()[0]
        index2 = ann_list[i+1].obs['batch'].unique()[0]
        cell_ids_1 = ann_list[i].obs.index.to_numpy()
        cell_ids_2 = ann_list[i+1].obs.index.to_numpy() 
        
        result = pd.DataFrame(solutions[(index1, index2)].transport_matrix)
        result.index = cell_ids_1
        result.columns = cell_ids_2

        matching_index = np.argmax(result.to_numpy(), axis=0)
        matching_cell_ids = pd.DataFrame({
            'id_1': cell_ids_1[matching_index],
            'id_2': cell_ids_2   
        })

        pi_list.append(result.to_numpy())
        pi_index_list.append(cell_ids_1)
        pi_column_list.append(cell_ids_2)
        matching_cell_ids_list.append(matching_cell_ids)
    
    adata = create_lightweight_adata(adata, config=config, pi_list=pi_list, pi_index_list=pi_index_list, pi_column_list=pi_column_list, matching_cell_ids_list=matching_cell_ids_list, args_dict=args_dict)
    
    return adata