import warnings
warnings.filterwarnings("ignore")

import os
import scanpy as sc
import numpy as np
import os
import pandas as pd
import seaborn as sns
import pandas as pd

from SPACEL import Scube

#################### Run SPACEL ####################
def run_SPACEL(ann_list, sample_list, output_path):
    i = 0
    for ann in ann_list:
        spatial_data = ann.obsm['spatial']
        spatial_df = pd.DataFrame(spatial_data, columns=['X', 'Y'], index=ann.obs_names)
        ann.obsm['spatial'] = spatial_df
        ann.obs['original_clusters'] = ann.obs['Ground Truth'].astype(str)
        ann.obsm['spatial']['Z'] = i
        i += 1
    adata = sc.concat(ann_list, index_unique=None) 
    df_path = os.path.join(output_path, 'alignment_'+str('_'.join(sample_list))+'.csv')
    Scube.align(ann_list,cluster_key='original_clusters',n_neighbors=15,knn_exclude_cutoff=25,p=3, n_threads=10, write_loc_path=df_path)
    alignment_df = pd.read_csv(df_path)
    alignment_df.rename(columns={"X": "x", "Y": "y"}, inplace=True)
    adata.obsm['spatial_aligned'] = np.array(alignment_df[['x', 'y']])
    adata.obsm['spatial'] = adata.obsm['spatial'][['X','Y']].values
    return adata    