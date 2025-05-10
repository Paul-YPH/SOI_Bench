import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import torch
import numpy as np
import os
from os.path import join as pj
import pandas as pd

import CAST
from CAST.utils import detect_highly_variable_genes
from CAST.visualize import plot_mid
from CAST.utils import extract_coords_exp
from CAST.models.model_GCNII import Args
from CAST import CAST_MARK
from CAST.visualize import kmeans_plot_multiple
from CAST.CAST_Stack import reg_params
from CAST import CAST_STACK

def make_unique(s):
    if s in counts:
        counts[s] += 1
    else:
        counts[s] = 0
    return s if counts[s] == 0 else f"{s}_{counts[s]}"

#################### Run CAST ####################
def run_CAST_2slices(ann_list,sample_list,output_path):
    ann_list.reverse()
    sample_list.reverse()
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    batch_key = 'batch'
    # Filter for highly variable genes
    adata.var['highly_variable'] = detect_highly_variable_genes(adata,batch_key=batch_key,n_top_genes=3000,count_layer='.X')
    
    if adata.var['highly_variable'].sum() == 0:
        print("No highly variable genes detected, using scanpy's method as fallback.")
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=min(3000, adata.shape[1]))
    
    adata = adata[:,adata.var['highly_variable']]
    adata.obs["x"] = pd.Series(adata.obsm["spatial"][:, 0], index=adata.obs.index)
    adata.obs["y"] = pd.Series(adata.obsm["spatial"][:, 1], index=adata.obs.index)
    coords_t = np.array(adata.obs[['x', 'y']])
    plot_mid(coords_t[adata.obs[batch_key] == sample_list[0]],
            coords_t[adata.obs[batch_key] == sample_list[1]],
            output_path=output_path,
            filename = 'Align_raw',
            title_t = [sample_list[1],
                        sample_list[0]],
            s_t = 8,scale_bar_t = None)
    coords_raw,exps = extract_coords_exp(adata, batch_key = 'batch', cols = ['x', 'y'], count_layer = '.X', data_format = 'norm1e4')
    # Perform CAST Mark
    args = Args(
        dataname='task1', # name of the dataset, used to save the log file
        gpu = 0, # gpu id, set to zero for single-GPU nodes
        epochs=400, # number of epochs for training
        lr1= 1e-3, # learning rate
        wd1= 0, # weight decay
        lambd= 1e-3, # lambda in the loss function, refer to online methods
        n_layers=2, # number of GCNII layers, more layers mean a deeper model, larger reception field, at a cost of VRAM usage and computation time
        der=0.5, # edge dropout rate in CCA-SSG
        dfr=0.3, # feature dropout rate in CCA-SSG
        use_encoder=True, # perform a single-layer dimension reduction before the GNNs, helps save VRAM and computation time if the gene panel is large
        encoder_dim=512, # encoder dimension, ignore if `use_encoder` set to `False`
    )
    # run CAST Mark
    embed_dict = CAST_MARK(coords_raw,exps,output_path,args = args,graph_strategy='delaunay')
    # plot the results
    kmeans_plot_multiple(embed_dict,sample_list,coords_raw,'demo1',output_path,k=20,dot_size = 10,minibatch=True)
    # set the parameters for CAST Stack
    query_sample = sample_list[1]
    params_dist = reg_params(dataname = query_sample,
                                gpu = 0 if torch.cuda.is_available() else -1,
                                #### Affine parameters
                                iterations=150,
                                dist_penalty1=0,
                                bleeding=500,
                                d_list = [3,2,1,1/2,1/3],
                                attention_params = [None,3,1,0],
                                #### FFD parameters
                                dist_penalty2 = [0],
                                alpha_basis_bs = [0],
                                meshsize = [8],
                                iterations_bs = [1],
                                attention_params_bs = [[None,3,1,0]],
                                mesh_weight = [None])
    # set the alpha basis for the affine transformation
    params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)
    # run CAST Stack
    coord_final = CAST_STACK(coords_raw,embed_dict,output_path,sample_list,params_dist,sub_node_idxs = None)
    return coord_final

def run_CAST(ann_list, sample_list, output_path):
    for ann in ann_list:
        ann.obsm['spatial_tmp'] = ann.obsm['spatial'].copy()
        ann.obsm['spatial_aligned'] = ann.obsm['spatial'].copy()

    for ii in range(len(ann_list) - 1):
        print(ii)
        coord_final = run_CAST_2slices(ann_list[ii:ii + 2], sample_list[ii:ii + 2], output_path)
        ann_list[ii + 1].obsm['spatial'] = coord_final[sample_list[ii:ii + 2][1]].cpu().numpy()
        ann_list[ii + 1].obsm['spatial_aligned'] = coord_final[sample_list[ii:ii + 2][1]].cpu().numpy()
    adata = sc.concat(ann_list, join='outer', index_unique=None)
    adata.obsm['spatial'] = adata.obsm['spatial_tmp'].copy()
    return adata