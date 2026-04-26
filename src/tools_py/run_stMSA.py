import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import numpy as np
import pandas as pd
import gc
from stMSA.utils import find_similar_index, coor_transform
from stMSA.alignment import get_transform
from stMSA.train_integrate import train_integration
from utils import get_ann_list, create_lightweight_adata,set_seed
from clustering import clustering

def run_stMSA(adata, **args_dict):
    seed = args_dict['seed']
    set_seed(seed)
    knn = args_dict['knn']
    clust = args_dict['clust']
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    adata = sc.concat(ann_list, join='inner', label='batch')
    
    # Embedding
    adata = train_integration(adata, knears=knn,dims=[512, min(30, adata.shape[1]-1, adata.shape[0]-1)])
    adata.obsm['integrated'] = adata.obsm['embedding']
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust)
    adata.obs['benchmark_cluster'] = adata.obs[clust].copy()
    
    for ann in ann_list:
        ann.obs[clust] = adata.obs[clust][ann.obs_names]
    
    # Matching
    pi_list = []
    pi_index_list = []
    pi_column_list = []
    matching_cell_ids_list = []
    batch_list = adata.obs['batch'].unique()
    
    for i in range(len(batch_list)-1):
        j = i+1
        src = adata[adata.obs['batch'] == batch_list[i]].copy()
        dst = adata[adata.obs['batch'] == batch_list[j]].copy()

        _, order = find_similar_index(
            src.obsm['embedding'],
            dst.obsm['embedding'],
        )

        matched_dst_indices = order[:, 0]

        cell_ids_src = src.obs_names.to_numpy()
        cell_ids_dst = dst.obs_names.to_numpy()
        result_np = np.zeros((len(cell_ids_src), len(cell_ids_dst)), dtype=float)

        src_indices = np.arange(len(cell_ids_src))
        result_np[src_indices, matched_dst_indices] = 1.0

        result = pd.DataFrame(result_np, index=cell_ids_src, columns=cell_ids_dst)
        matching_cell_ids = pd.DataFrame({
            'id_1': cell_ids_src,                         
            'id_2': cell_ids_dst[matched_dst_indices]   
        })

        pi_list.append(result.to_numpy())
        pi_index_list.append(cell_ids_src)
        pi_column_list.append(cell_ids_dst)
        matching_cell_ids_list.append(matching_cell_ids)
        
    # Alignment
    dst_id = 0
    src_id_list = list(range(1, len(batch_list)))
    
    common_labels = set(ann_list[0].obs['mclust'].unique())
    for i in range(1, len(ann_list)):
        current_labels = set(ann_list[i].obs['mclust'].unique())
        common_labels = common_labels.intersection(current_labels)

    if len(common_labels) > 0:
        common_labels_list = list(common_labels)
        common_labels_list.sort(key=lambda x: int(x))
        pyramidal_label_id = common_labels_list[0]
        print('pyramidal_label_id:', pyramidal_label_id)
    else:
        print("No common 'mclust' cluster found across ALL AnnData objects in ann_list! Using all locations.")
        for ann in ann_list:
            ann.obs['mclust'] = '0'
        pyramidal_label_id = '0'
        print('pyramidal_label_id:', pyramidal_label_id)

    M = get_transform(ann_list, dst_id, src_id_list, pyramidal_label_id)
    coor_dict = {src_id: coor_transform(ann_list[src_id].obsm['spatial'], M[src_id])[:2, :].T for src_id in src_id_list}
    
    adata.obsm['spatial_aligned'] = adata.obsm['spatial'].copy()
    adata.obs['mclust'] = adata.obs['benchmark_cluster'].copy()
    for src_id in src_id_list:
        idx = adata.obs_names.get_indexer(ann_list[src_id].obs_names)
        adata.obsm['spatial_aligned'][idx] = coor_dict[src_id]
    
    adata = create_lightweight_adata(adata, config=config, pi_list=pi_list, pi_index_list=pi_index_list, pi_column_list=pi_column_list, matching_cell_ids_list=matching_cell_ids_list, args_dict=args_dict)
    
    return adata