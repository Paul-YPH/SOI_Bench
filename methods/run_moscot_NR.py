
import warnings
warnings.filterwarnings("ignore")

import moscot as mt
from moscot import datasets
from moscot.problems.space import AlignmentProblem

import scanpy as sc
import numpy as np
import pandas as pd
import squidpy as sq

#################### Run moscot ####################

def run_moscot_NR(ann_list):
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    
    sample = []
    for ann in ann_list:
        batch_values = ann.obs['batch'].unique()[0]
        sample.append(batch_values)    
    adata.obs['batch'] = pd.Categorical(adata.obs['batch'], categories=sample, ordered=True)

    ap = AlignmentProblem(adata=adata)
    ap = ap.prepare(batch_key="batch", policy="sequential")
    ap = ap.solve(alpha=0.5) ###Spateo
    ap.align(reference=ann_list[0].obs['batch'].unique()[0], key_added="spatial_aligned")
    
    solutions = ap.solutions
    pis = list(solutions.values())
    for i in range(len(pis)):
        pis[i] = list(solutions.values())[i].transport_matrix
        
    pi_list = []
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
            'cell_id_1': cell_ids_1[matching_index],
            'cell_id_2': cell_ids_2   
        })

        pi_list.append(result)
        matching_cell_ids_list.append(matching_cell_ids)
        
    return adata, pi_list, matching_cell_ids_list