import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse

import pandas as pd
import numpy as np
import scanpy as sc
import ot

import time
import resource
import torch

import paste as pst

#################### Run PASTE2 ####################
def run_PASTE(ann_list, sample_list):
    pi_list = []
    matching_cell_ids_list = []
    for i in range(len(sample_list)-1):
        j = i+1
        adata1 = ann_list[i]
        adata2 = ann_list[j]
        pi0 = pst.match_spots_using_spatial_heuristic(adata1.obsm['spatial'],adata2.obsm['spatial'],use_ot=True)
        pi12 = pst.pairwise_align(adata1, adata2, G_init=pi0, norm=True, backend=ot.backend.TorchBackend(), use_gpu = True)

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
    new_slices = pst.stack_slices_pairwise(ann_list, pi_array_list)
    for i in range(0, len(new_slices)):
        new_slices[i].obsm['spatial_aligned'] = new_slices[i].obsm['spatial']
        new_slices[i].obsm['spatial'] = ann_list[i].obsm['spatial']
    adata = sc.concat(new_slices, join='inner', index_unique=None)

    print("adata shape:", adata.shape)
    print("pi_list length:", pi_list[0].shape)
    return adata, pi_list, matching_cell_ids_list


# ['151507','151508'] Not ok
# ['151673','151674'] ok