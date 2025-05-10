import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import scanpy as sc
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset

from gpsa import VariationalGPSA
from gpsa import rbf_kernel

def run_GPSA_2slices(ann_list):
    N_GENES = 10
    N_SAMPLES = None

    n_spatial_dims = 2
    n_views = 2
    m_G = 200
    
    m_X_per_view = min(ann_list[0].X.shape[0], 200)
    m_X_per_view = min(ann_list[1].X.shape[0], m_X_per_view)
    
    m_G = min(ann_list[0].X.shape[0], 200)
    m_G = min(ann_list[1].X.shape[0], m_G)

    N_LATENT_GPS = {"expression": None}

    N_EPOCHS =  5000
    PRINT_EVERY = 50
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def scale_spatial_coords(X, max_val=10.0):
        X = X - X.min(0)
        X = X / X.max(0)
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

        # sc.pp.filter_cells(adata, min_counts=5000)
        # sc.pp.filter_cells(adata, max_counts=35000)
        # adata = adata[adata.obs["pct_counts_mt"] < 20]
        # sc.pp.filter_genes(adata, min_cells=0)

        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        print(adata.shape[1])
        if adata.shape[1]>n_top_genes:  
            sc.pp.highly_variable_genes(
                adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
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
        process_data(ann, n_top_genes=min(ann.X.shape[1], 200))
        sc.pp.filter_cells(ann, min_genes=1)
        sc.pp.filter_genes(ann, min_cells=1)
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

    import scipy.sparse as sp
    def to_dense(matrix):
        """Convert sparse matrix to dense if needed."""
        return matrix.todense() if sp.issparse(matrix) else matrix

    Y1 = to_dense(ann_list[0].X)
    Y2 = to_dense(ann_list[1].X)

    X1 = scale_spatial_coords(X1)
    X2 = scale_spatial_coords(X2)
    Y1 = (Y1 - Y1.mean(0)) / Y1.std(0)
    Y2 = (Y2 - Y2.mean(0)) / Y2.std(0)
    X = np.concatenate([X1, X2])
    Y = np.concatenate([Y1, Y2])
    n_outputs = Y.shape[1]
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
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

def run_GPSA(ann_list):
    
    for ann in ann_list:
        ann.obsm['spatial_tmp'] = ann.obsm['spatial'].copy()
        ann.obsm['spatial_aligned'] = ann.obsm['spatial'].copy()
    
    for ii in range(len(ann_list) - 1):
        aligned_ann = run_GPSA_2slices(ann_list[ii:ii + 2])

        for idx in range(ii, ii + 2):
            ann_list[idx].obsm['spatial'] = aligned_ann[ann_list[idx].obs_names].obsm['spatial_aligned']
            ann_list[idx].obsm['spatial_aligned'] = aligned_ann[ann_list[idx].obs_names].obsm['spatial_aligned']

    adata = sc.concat(ann_list, join='inner', index_unique=None)
    adata.obsm['spatial'] = adata.obsm['spatial_tmp'].copy()
    adata.X = adata.layers['counts']
    return adata
