import warnings
warnings.filterwarnings("ignore")

import os
import scanpy as sc
from spatialign import Spatialign
import pandas as pd
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed
from clustering import clustering

def run_spatiAlign(adata, output_path, **args_dict):
    sample_list = adata.uns['config']['sample_list']
    seed = args_dict['seed']
    set_seed(seed)
    knn = args_dict['knn']
    clust = args_dict['clust']
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    sample_list_path = []
    for ann in ann_list:
        ann_path = os.path.join(output_path, ann.obs['batch'].unique()[0]+'_spatialign.h5ad')
        sample_list_path.append(ann_path)
        ann.write_h5ad(ann_path)
    # Perform spatiAlign
    model = Spatialign(
        *sample_list_path,
        min_genes = 1,
        min_cells = 1,
        batch_key='batch',
        is_norm_log=True,
        is_scale=False,
        n_neigh=knn,
        is_undirected=True,
        seed=seed,
        gpu=0,
        save_path=output_path,
        is_verbose=False
    )
    model.train()
    model.alignment()

    for n in range(len(sample_list)):
        tmp = sc.read_h5ad(os.path.join(output_path,'res',"correct_data"+str(n)+".h5ad"))
        tmp.obs['batch'] = sample_list[n]
        common_cells = ann_list[n].obs_names.intersection(tmp.obs_names)
        ann_list[n] = ann_list[n][common_cells]
        tmp = tmp[common_cells]
        ann_list[n].obsm['correct'] = tmp.obsm['correct']
        del tmp
        gc.collect()
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    del ann_list
    gc.collect()
    
    adata.obsm['integrated'] = adata.obsm['correct']
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust)
    res_dir = os.path.join(output_path, 'res')
    if os.path.exists(res_dir):
        for file_name in os.listdir(res_dir):
            file_path = os.path.join(res_dir, file_name)
            if file_name.endswith('.h5ad'):
                os.remove(file_path)
    adata.obs['benchmark_cluster'] = adata.obs[clust]
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata