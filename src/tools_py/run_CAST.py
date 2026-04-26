import warnings
warnings.filterwarnings("ignore")
import scanpy as sc
import torch
import numpy as np
import pandas as pd
import gc
from CAST.utils import detect_highly_variable_genes, extract_coords_exp
from CAST.models.model_GCNII import Args
from CAST import CAST_MARK, CAST_STACK
from CAST.CAST_Stack import reg_params
from utils import get_ann_list, create_lightweight_adata,set_seed

def run_CAST_2slices(ann_list, sample_list, output_path, n_top_genes=3000):
    ann_list = list(ann_list)[::-1]
    sample_list = list(sample_list)[::-1]

    adata = sc.concat(ann_list, join='inner', index_unique=None)
    del ann_list
    gc.collect()
    batch_key = 'batch'
    adata.var['highly_variable'] = detect_highly_variable_genes(
        adata, batch_key=batch_key, n_top_genes=min(n_top_genes, min(adata.shape)), count_layer='.X'
    )
    adata = adata[:, adata.var['highly_variable']]
    n_genes_used = int(adata.n_vars)

    adata.obs["x"] = pd.Series(adata.obsm["spatial"][:, 0], index=adata.obs.index)
    adata.obs["y"] = pd.Series(adata.obsm["spatial"][:, 1], index=adata.obs.index)
    coords_raw, exps = extract_coords_exp(adata, batch_key='batch', cols=['x','y'], count_layer='.X', data_format='norm1e4')

    args = Args(dataname='task1', gpu=0 if torch.cuda.is_available() else -1,
                epochs=400, lr1=1e-3, wd1=0, lambd=1e-3, n_layers=2,
                der=0.5, dfr=0.3, use_encoder=True, encoder_dim=512)

    embed_dict = CAST_MARK(coords_raw, exps, output_path, args=args, graph_strategy='delaunay')

    query_sample = sample_list[1]

    def auto_set_bleeding(coords_raw, d_list=[3,2,1,0.5,0.333]):
        coords_list = list(coords_raw.values())
        c0 = np.array(coords_list[0])
        c1 = np.array(coords_list[1])
        
        range0 = (c0.max() - c0.min())
        range1 = (c1.max() - c1.min())
        center_offset = np.abs(c0.mean(0) - c1.mean(0)).max()
        
        max_range = max(range0, range1)
        print(f"Slice ranges: {range0:.1f}, {range1:.1f}, center_offset: {center_offset:.1f}")
        
        bleeding = int(max(max_range * 0.5, center_offset * 1.5, 500))
        bleeding = min(bleeding, 8000)
        
        print(f"→ Auto-set bleeding = {bleeding}")
        return bleeding

    bleeding = auto_set_bleeding(coords_raw, d_list=[3,2,1,0.5,0.333])
    
    params_dist = reg_params(dataname=query_sample, gpu=0 if torch.cuda.is_available() else -1,
        iterations=150, dist_penalty1=0, bleeding=bleeding, d_list=[3,2,1,0.5,0.333],
        attention_params=[None,3,1,0], dist_penalty2=[0], alpha_basis_bs=[0],
        meshsize=[8], iterations_bs=[1], attention_params_bs=[[None,3,1,0]], mesh_weight=[None])
    params_dist.alpha_basis = torch.Tensor([1/1000,1/1000,1/50,5,5]).reshape(5,1).to(params_dist.device)


    import CAST.CAST_Stack as cs
    original_J_cal = cs.J_cal

    def safe_J_cal(coords_q, coords_r, cov_mat, bleeding=10, dist_penalty=0, attention_params=[None,3,1,0]):
        bleeding_x = coords_q[:, 0].min() - bleeding, coords_q[:, 0].max() + bleeding
        bleeding_y = coords_q[:, 1].min() - bleeding, coords_q[:, 1].max() + bleeding
        sub_ind = (
            (coords_r[:, 0] > bleeding_x[0]) & (coords_r[:, 0] < bleeding_x[1]) &
            (coords_r[:, 1] > bleeding_y[0]) & (coords_r[:, 1] < bleeding_y[1])
        )
        if sub_ind.sum() == 0:
            q_min = coords_q.min(0).values
            q_max = coords_q.max(0).values
            r_min = coords_r.min(0).values
            r_max = coords_r.max(0).values
            needed_bleeding = max(
                (q_min - r_max).abs().max().item(),
                (q_max - r_min).abs().max().item()
            )
            bleeding = int(needed_bleeding * 1.2)
            print(f"  ⚠️  bleeding expanded to {bleeding} to cover coords_r")
        return original_J_cal(coords_q, coords_r, cov_mat, bleeding, dist_penalty, attention_params)

    cs.J_cal = safe_J_cal

    print(f"coords_raw shapes: {[c.shape for c in coords_raw.values()]}")
    print(f"embed_dict shapes: {[e.shape for e in embed_dict.values()]}")

    coord_final = CAST_STACK(coords_raw, embed_dict, output_path, sample_list, params_dist, sub_node_idxs=None)

    cs.J_cal = original_J_cal

    return coord_final, n_genes_used

def run_CAST(adata, output_path, **args_dict):
    seed = args_dict['seed']
    set_seed(seed)
    
    sample_list = adata.uns['config']['sample_list']
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    for ann in ann_list:
        ann.obsm['spatial_tmp'] = ann.obsm['spatial'].copy()
        ann.obsm['spatial_aligned'] = ann.obsm['spatial'].copy()
        sc.pp.filter_genes(ann, min_cells=1)
        sc.pp.filter_cells(ann, min_genes=1)

    for ii in range(len(ann_list) - 1):
        coord_final = run_CAST_2slices(
            ann_list[ii:ii+2], sample_list[ii:ii+2], output_path
        )
        coords_dict = coord_final[0]
        coord_item = coords_dict[sample_list[ii+1]]
        if hasattr(coord_item, "cpu"):  # torch.Tensor
            ann_list[ii+1].obsm["spatial"] = coord_item.cpu().numpy()
            ann_list[ii+1].obsm["spatial_aligned"] = coord_item.cpu().numpy()
        else:  # numpy array
            ann_list[ii+1].obsm["spatial"] = np.array(coord_item)
            ann_list[ii+1].obsm["spatial_aligned"] = np.array(coord_item)

    adata = sc.concat(ann_list, join='inner', index_unique=None)
    del ann_list
    gc.collect()
    adata.obsm['spatial'] = adata.obsm['spatial_tmp'].copy()
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata