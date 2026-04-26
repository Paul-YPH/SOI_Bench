import warnings
warnings.filterwarnings("ignore")

import os
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import pandas as pd
import gc
from SPACEL import Scube
from utils import get_ann_list, create_lightweight_adata,set_seed

def run_SPACEL(adata, output_path, **args_dict):
    seed = args_dict['seed']
    knn = args_dict['knn']
    set_seed(seed)
    
    sample_list = adata.uns['config']['sample_list']
    ann_list, config = get_ann_list(adata)

    obs_info = ann_list[0].obs

    if 'data_id' in obs_info.columns:
        data_id = str(obs_info['data_id'].unique()[0])
    elif 'technology' in obs_info.columns:
        data_id = str(obs_info['technology'].unique()[0])
    else:
        data_id = "unknown_sample"

    del adata,obs_info
    gc.collect()
    
    i = 0
    for ann in ann_list:
        spatial_data = ann.obsm['spatial']
        spatial_df = pd.DataFrame(spatial_data, columns=['X', 'Y'], index=ann.obs_names)
        ann.obsm['spatial'] = spatial_df
        ann.obs['original_clusters'] = ann.obs['Ground Truth'].astype(str)  ###
        ann.obsm['spatial']['Z'] = i
        i += 1
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_name = f"alignment_{data_id}_{knn}_{seed}.csv"
    df_path = os.path.join(output_path, file_name)
    Scube.align(ann_list,cluster_key='original_clusters',n_neighbors=knn,knn_exclude_cutoff=10,p=1,  write_loc_path=df_path)
    alignment_df = pd.read_csv(df_path)
    alignment_df.rename(columns={"X": "x", "Y": "y"}, inplace=True)
    
    adata = sc.concat(ann_list, index_unique=None) 
    del ann_list
    gc.collect()
    
    adata.obsm['spatial_aligned'] = np.array(alignment_df[['x', 'y']])
    adata.obsm['spatial'] = adata.obsm['spatial'][['X','Y']].values
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata    