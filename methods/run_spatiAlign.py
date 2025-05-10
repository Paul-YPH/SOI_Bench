import warnings
warnings.filterwarnings("ignore")

import os
import sys

import scanpy as sc
from spatialign import Spatialign
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__)) 
benchmark_dir = os.path.dirname(current_dir)  
sys.path.append(benchmark_dir)
from metrics.utils import *

#################### Run spatiAlign ####################
def run_spatiAlign(ann_list, sample_list, output_path, cluster_option,sample_list_path):
    
    print(sample_list_path)
    # Perform spatiAlign
    model = Spatialign(
        *sample_list_path,
        min_genes = 0,
        min_cells = 0,
        batch_key='batch',
        is_norm_log=True,
        is_scale=False,
        n_neigh=15,
        is_undirected=True,
        latent_dims=100,
        seed=42,
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
    adata = sc.concat(ann_list, join='inner', index_unique=None)
    adata.obsm['integrated'] = adata.obsm['correct']
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=cluster_option, start=0.01, end=2.0, increment=0.01)
    
    # print('### Cleaning up intermediate files...')
    # for file_path in filename_list:
    #     if os.path.exists(file_path):
    #         os.remove(file_path)
    res_dir = os.path.join(output_path, 'res')
    if os.path.exists(res_dir):
        for file_name in os.listdir(res_dir):
            file_path = os.path.join(res_dir, file_name)
            if file_name.endswith('.h5ad'):
                os.remove(file_path)
    return adata