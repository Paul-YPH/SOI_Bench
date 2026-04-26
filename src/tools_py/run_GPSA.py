import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import scanpy as sc
import gc
from gpsa import VariationalGPSA
from gpsa import rbf_kernel
from utils import get_ann_list, create_lightweight_adata,set_seed,filter_common_genes

def recover_gpsa_scale(adata, ref_batch='151673_A'):
    if 'spatial_original' not in adata.obsm or 'spatial_aligned' not in adata.obsm:
        return adata

    available_batches = adata.obs['batch'].unique()
    actual_ref = ref_batch if ref_batch in available_batches else available_batches[0]

    ref_mask = adata.obs['batch'] == actual_ref
    orig_ref = adata.obsm['spatial_original'][ref_mask]
    x_min = orig_ref.min(0)
    x_max_range = orig_ref.max(0) - x_min
    x_max_range[x_max_range == 0] = 1.0

    current_max_val = adata.obsm['spatial_aligned'].max()

    if current_max_val == 0:
        return adata

    adata.obsm['spatial_aligned_scaled_bak'] = adata.obsm['spatial_aligned'].copy()
    recovered_coords = (adata.obsm['spatial_aligned'] / current_max_val) * x_max_range + x_min
    adata.obsm['spatial_aligned'] = recovered_coords

    return adata

def run_GPSA_2slices(ann_list):
    N_GENES = 10
    N_SAMPLES = None

    n_spatial_dims = 2
    n_views = 2

    m_G = 100
    lr = 1e-3
    max_val = 10

    if 'data_id' in ann_list[0].obs.columns:
       data_id = ann_list[0].obs['data_id'].unique()[0]
       if data_id == 'SD6':
        m_G = 500
       elif data_id in ['SD29','SD40','SD43']:
        lr = 1e-4


    m_X_per_view = min(ann_list[0].X.shape[0], m_G)
    m_X_per_view = min(ann_list[1].X.shape[0], m_X_per_view)
    
    m_G = min(ann_list[0].X.shape[0], m_G)
    m_G = min(ann_list[1].X.shape[0], m_G)

    N_LATENT_GPS = {"expression": None}

    N_EPOCHS =  5000
    PRINT_EVERY = 50
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def scale_spatial_coords(X, max_val=10.0):
        X = X - X.min(0)
        x_range = X.max(0)
        x_range[x_range == 0] = 1.0
        X = X / x_range
        return X * max_val

    def process_data(adata, n_top_genes=2000):
        adata.var_names_make_unique()
        
        if adata.obs['species'].unique() == 'Human':
            adata.var["mt"] = adata.var_names.str.startswith("MT-")
        else:
            adata.var["mt"] = adata.var_names.str.startswith("mt-")
        if adata.var["mt"].sum() == 0 or 'mt' not in adata.var.columns:
            sc.pp.calculate_qc_metrics(adata, inplace=True,percent_top=[min(adata.X.shape[1],n_top_genes)])
        else:
            sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=[min(adata.X.shape[1],n_top_genes)])

        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        if adata.shape[1]>n_top_genes:  
            sc.pp.highly_variable_genes(
                adata, flavor="seurat", n_top_genes=n_top_genes
            )
        else:
            adata.var['highly_variable'] = True
        return adata

    def train(model, loss_fn, optimizer):
        model.train()

        # Forward pass
        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns, S=5
        )

        # Compute loss
        loss = loss_fn(data_dict, F_samples)

        # Compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), G_means

    for i, ann in enumerate(ann_list):
        process_data(ann, n_top_genes=min(ann.X.shape[1], m_G))
        sc.pp.filter_cells(ann, min_genes=1)
        sc.pp.filter_genes(ann, min_cells=1)
    
    hvg_list = [set(ann.var_names[ann.var['highly_variable']]) for ann in ann_list]
    common_hvg = list(set.intersection(*hvg_list))
    if len(common_hvg) >= 10:
        ann_list = [ann[:, common_hvg].copy() for ann in ann_list]
        adata = sc.concat(ann_list, join='inner', index_unique=None)
    else:
        ann_list = filter_common_genes(ann_list)
        adata = sc.concat(ann_list, join='inner', index_unique=None)
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=10, 
            flavor="seurat", 
            batch_key='batch'
        )
        global_hvg = adata.var_names[adata.var['highly_variable']].tolist()
        adata = adata[:, global_hvg].copy()
        ann_list = [ann[:, global_hvg].copy() for ann in ann_list]


    ann_list = filter_common_genes(ann_list)
    adata = sc.concat(ann_list,join='inner', index_unique=None)  
    # if N_SAMPLES is not None:
    #     rand_idx = np.random.choice(
    #         np.arange(ann_list[0].shape[0]), size=N_SAMPLES, replace=False
    #     )
    #     ann_list[0] = ann_list[0][rand_idx]
    #     rand_idx = np.random.choice(
    #         np.arange(ann_list[1].shape[0]), size=N_SAMPLES, replace=False
    #     )
    #     ann_list[1] = ann_list[1][rand_idx]

    # all_slices = anndata.concat([data_slice1, data_slice2])
    n_samples_list = [ann_list[0].shape[0], ann_list[1].shape[0]]
    view_idx = [
        np.arange(ann_list[0].shape[0]),
        np.arange(ann_list[0].shape[0], ann_list[0].shape[0] + ann_list[1].shape[0]),
    ]
    X1 = ann_list[0].obsm["spatial"]
    X2 = ann_list[1].obsm["spatial"]

    def handle_duplicates(X, eps=1e-4):
        if X.shape[0] != np.unique(X, axis=0).shape[0]:
            return X + np.random.normal(0, eps, size=X.shape)
        return X

    X1 = handle_duplicates(X1)
    X2 = handle_duplicates(X2)

    import scipy.sparse as sp
    def to_dense(matrix):
        """Convert sparse matrix to dense if needed."""
        return matrix.todense() if sp.issparse(matrix) else matrix

    Y1 = to_dense(ann_list[0].X)
    Y2 = to_dense(ann_list[1].X)

    X1 = scale_spatial_coords(X1,max_val)
    X2 = scale_spatial_coords(X2,max_val)

    is_sd = False
    if 'data_id' in ann_list[0].obs.columns:
        data_id = ann_list[0].obs['data_id'].unique()[0]
        is_sd = str(data_id).startswith('SD')

    if is_sd:
        Y1 = np.array(Y1)
        Y2 = np.array(Y2)
        Y1_std = Y1.std(0)
        Y1_std[Y1_std < 1e-8] = 1.0
        Y2_std = Y2.std(0)
        Y2_std[Y2_std < 1e-8] = 1.0
        Y1 = (Y1 - Y1.mean(0)) / Y1_std
        Y2 = (Y2 - Y2.mean(0)) / Y2_std
        Y1 = np.clip(Y1, -10, 10)
        Y2 = np.clip(Y2, -10, 10)
    else:
        Y1_std = np.array(Y1.std(0))
        Y1_std[Y1_std == 0] = 1.0
        Y2_std = np.array(Y2.std(0))
        Y2_std[Y2_std == 0] = 1.0
        Y1 = (Y1 - Y1.mean(0)) / Y1_std
        Y2 = (Y2 - Y2.mean(0)) / Y2_std
    
    X = np.concatenate([X1, X2])
    Y = np.concatenate([Y1, Y2])
    print("Contains NaN?", np.isnan(np.array(Y1)).any())
    print("Contains NaN?", np.isnan(np.array(Y2)).any())
    
    x = torch.from_numpy(X).float().clone().to(device)
    y = torch.from_numpy(Y).float().clone().to(device)
    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }
    model = VariationalGPSA(
        data_dict,
        n_spatial_dims=n_spatial_dims,
        m_X_per_view=m_X_per_view,
        m_G=m_G,
        data_init=True,
        minmax_init=False,
        grid_init=False,
        n_latent_gps=N_LATENT_GPS,
        mean_function="identity_fixed",
        kernel_func_warp=rbf_kernel,
        kernel_func_data=rbf_kernel,
        # fixed_warp_kernel_variances=np.ones(n_views) * 1.,
        # fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
        fixed_view_idx=0,
    ).to(device)
    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    for t in tqdm(range(N_EPOCHS), desc="Training Progress"):
        loss, G_means = train(model, model.loss_fn, optimizer)
        curr_aligned_coords = G_means["expression"].detach().cpu().numpy()
    print("Done!")
    tmp = pd.DataFrame(
        {
            "aligned_x": curr_aligned_coords.T[0],
            "aligned_y": curr_aligned_coords.T[1],
        },
    )
    tmp.index = adata.obs_names
    adata.obsm['spatial_aligned'] = tmp.values
    return adata

def run_GPSA(adata, **args_dict):
    seed = args_dict['seed']
    set_seed(seed)
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    for ann in ann_list:
        ann.obsm['spatial_tmp'] = ann.obsm['spatial'].copy()
        ann.obsm['spatial_aligned'] = ann.obsm['spatial'].copy()
    
    for ii in range(len(ann_list) - 1):
        aligned_ann = run_GPSA_2slices(ann_list[ii:ii + 2])
        aligned_ann = recover_gpsa_scale(
                    aligned_ann,
                    ref_batch=ann_list[ii].obs['batch'].unique()[0]
                )
        for idx in range(ii, ii + 2):
            ann_list[idx].obsm['spatial'] = aligned_ann[ann_list[idx].obs_names].obsm['spatial_aligned']
            ann_list[idx].obsm['spatial_aligned'] = aligned_ann[ann_list[idx].obs_names].obsm['spatial_aligned']

    adata = sc.concat(ann_list, join='inner', index_unique=None)
    del ann_list
    gc.collect()
    
    adata.obsm['spatial'] = adata.obsm['spatial_tmp'].copy()
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata
