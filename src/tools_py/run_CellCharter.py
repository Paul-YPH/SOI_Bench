import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import squidpy as sq
import cellcharter as cc
import pandas as pd
import scanpy as sc
import scvi
import numpy as np
import gc
from lightning.pytorch import seed_everything
from utils import get_ann_list, create_lightweight_adata,set_seed

def run_CellCharter(adata, **args_dict):
    seed = args_dict['seed']
    knn = args_dict['knn']
    set_seed(seed)
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    seed_everything(seed)
    scvi.settings.seed = seed
    adata = ad.concat(ann_list, join='inner', index_unique=None)
    del ann_list
    gc.collect()
    
    adata.uns['spatial_fov'] = {s: {} for s in adata.obs['batch'].unique()}
    adata.obs['sample'] = pd.Categorical(adata.obs['batch'])
    
    adata.layers['counts'] = adata.X.copy()
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    scvi.model.SCVI.setup_anndata(
        adata, 
        layer="counts", 
        batch_key='sample',
    )
    model = scvi.model.SCVI(adata)
    model.train(early_stopping=True, enable_progress_bar=True)
    adata.obsm['X_scVI'] = model.get_latent_representation(adata).astype(np.float32)
    
    sq.gr.spatial_neighbors(adata, library_key='sample', coord_type='generic', delaunay=True, spatial_key='spatial', n_neighs = knn)
    cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep='X_scVI', out_key='X_cellcharter', sample_key='sample')
    autok = cc.tl.Cluster(
        n_clusters=len(adata.obs['Ground Truth'].unique()),
    trainer_params=dict(accelerator='gpu', devices=1)
)
    autok.fit(adata, use_rep='X_cellcharter')
    adata.obs['benchmark_cluster'] = autok.predict(adata, use_rep='X_cellcharter')
    adata.obsm['integrated'] = adata.obsm['X_cellcharter']
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata