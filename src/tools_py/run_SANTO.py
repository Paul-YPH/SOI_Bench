import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import scanpy as sc
import torch
import pandas as pd
import numpy as np
import santo
import easydict
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed

def run_SANTO(adata, **args_dict):
    seed = args_dict['seed']
    knn = args_dict['knn']
    set_seed(seed)
    
    overlap_list = adata.uns['config']['overlap']
    diff_omics = adata.uns['config']['multiomics_cross_slice']
    ann_list, config = get_ann_list(adata)
    del adata
    if overlap_list is True:
        align_mode = 'stitch'
    else:
        align_mode = 'align'
    
    for ann in ann_list:
        sc.pp.normalize_total(ann)
        ann.obsm['spatial'] = ann.obsm['spatial'].astype(np.float32)
    adata = sc.concat(ann_list, index_unique=None)  
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=1)
    del ann_list
    gc.collect()
    
    args = easydict.EasyDict({})
    args.epochs = 40
    args.lr = 0.001
    args.k = knn
    args.alpha = 0.9 # weight of transcriptional loss
    args.diff_omics = diff_omics # whether to use different omics data
    args.mode = align_mode # Choose the mode among 'align', 'stitch' and None
    args.dimension = 2  # choose the dimension of coordinates (2 or 3)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # choose the device
    
    slice_ids = np.unique(adata.obs['batch'])
    cor = np.array(adata.obsm['spatial'])
    #slice_ids = slice_ids[::-1]
    for idx in tqdm(range(len(slice_ids) - 1, 0, -1)):
        src = adata[adata.obs['batch'] == slice_ids[idx], :].copy()
        tgt = adata[adata.obs['batch'] == slice_ids[idx - 1], :].copy()

        src.obsm['spatial'] = cor[adata.obs['batch'] == slice_ids[idx], :]
        tgt.obsm['spatial'] = cor[adata.obs['batch'] == slice_ids[idx - 1], :]

        aligned_src_cor, trans_dict = santo.santo(src, tgt, args)
        src_mask = adata.obs['batch'] == slice_ids[idx]
        cor[src_mask, :] = (
            np.dot(cor[src_mask, :], trans_dict['coarse_R_ab'].T)
            + trans_dict['coarse_T_ab']
        )
        mask = np.isin(adata.obs['batch'], slice_ids[idx:])
        cor[mask, :] = np.dot(cor[mask, :], trans_dict['fine_R_ab'].T) + trans_dict['fine_T_ab']

        del src, tgt, aligned_src_cor, trans_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    adata.obsm['spatial_aligned'] = cor    
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata
    
    