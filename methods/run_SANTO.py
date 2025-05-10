import warnings
warnings.filterwarnings("ignore")

import os
import sys

import easydict
from tqdm import tqdm
import scanpy as sc
import torch
import pandas as pd
import numpy as np

import santo

#################### Run SANTO ####################
def run_SANTO(ann_list,overlap_list):
    if overlap_list is not None:
        align_mode = 'stitch'
    else:
        align_mode = 'align'
    
    for ann in ann_list:
        ann.layers['counts'] = ann.X.copy()
        sc.pp.normalize_total(ann)
    adata = sc.concat(ann_list, index_unique=None)  

    num_cells, num_genes = ann_list[0].shape
    same_cells = all(ann.shape[0] == num_cells for ann in ann_list)
    same_genes = all(ann.shape[1] == num_genes for ann in ann_list)
    
    first_techs = [ann.obs['technology'].iloc[0] for ann in ann_list if 'technology' in ann.obs]
    same_tech = all(tech == first_techs[0] for tech in first_techs)
    
    args = easydict.EasyDict({})
    args.epochs = 1000
    
    if same_cells or same_genes or not same_tech:
        args.lr = 0.00005
    else:
        args.lr = 0.0001
        
    args.k = 10
    args.alpha = 0.9 # weight of transcriptional loss
    args.diff_omics = False # whether to use different omics data
    args.mode = align_mode # Choose the mode among 'align', 'stitch' and None
    args.dimension = 2  # choose the dimension of coordinates (2 or 3)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # choose the device
    
    slice_ids = np.unique(adata.obs['batch'])
    #slice_ids = slice_ids[::-1]
    cor = np.array(adata.obsm['spatial'])
    for id in tqdm(range(len(slice_ids) - 1, 0, -1)):  
        src = adata[adata.obs['batch'] == slice_ids[id], :].copy()
        tgt = adata[adata.obs['batch'] == slice_ids[id - 1], :].copy()
        src.obs.isna().sum()
        tgt.obs.isna().sum()
        aligned_src_cor, trans_dict = santo.santo(src, tgt, args)
        print(trans_dict)
        cor[np.isin(adata.obs['batch'], slice_ids[id:]), :] = (
            np.dot(
                np.dot(cor[np.isin(adata.obs['batch'], slice_ids[id:]), :], trans_dict['coarse_R_ab'].T) 
                + trans_dict['coarse_T_ab'],
                trans_dict['fine_R_ab'].T
            )
            + trans_dict['fine_T_ab']
        )
    adata.obsm['spatial_aligned'] = cor
    adata.X = adata.layers['counts']
    return adata
    # # fine_angle = np.degrees(np.arctan2(fine_R_ab[1, 0], fine_R_ab[0, 0]))
    # # adata.obs['angle'] = fine_angle
    
    