import warnings
warnings.filterwarnings("ignore")

import os
from tacos import process_adata,integrate_datasets
from tacos import Tacos
from utils import get_ann_list, create_lightweight_adata,set_seed
from clustering import clustering
import gc

def run_Tacos(adata, output_path, **args_dict):
    clust = args_dict['clust']
    seed = args_dict['seed']
    set_seed(seed)
    
    adata_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
    processed_list = []
    load_list = []
    for i in range(len(adata_list)):
        adata = adata_list[i]
        section_id = adata.obs['batch'].unique()[0]
        adata.var_names_make_unique(join="++")
        adata_list[i] = process_adata(
            adata,
            marker_genes=[],
            min_genes=1,
            min_cells=1,
            n_top_genes=min(3000, adata.shape[1], adata.shape[0])
        )
        load_list.append(section_id)
    adata = integrate_datasets(adata_list, load_list)
    
    del adata_list, load_list
    gc.collect()
    
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
    tacos = Tacos(adata,latent_dim=min(adata.shape[0],adata.shape[1],50),gpu=0,check_detect=True,path=output_path)
    train_args = { #default para
        'epoch': 1500,
        'base_w':1.0,
        'base':'csgcl',
        'spatial_w':1.5,
        'cross_w':1.0,
        'recon_w' :0.0,
        'lr':1e-3,
        'max_patience':100,
        'min_stop':500,
        'cpu':1,
        'k':50,
        'update_mnn':100,
        'save_inter':500,
        'csgcl_arg':{
            'ced_drop_rate_1' : 0.2,
            'ced_drop_rate_2' : 0.7,
            'cav_drop_rate_1' : 0.1,
            'cav_drop_rate_2' : 0.2,
            't0':5,
            'gamma':1
            },
        'cross_arg':{
            'alpha':1.0,
            # 'negative':True

        },
        'spatial_arg':{
            'regularization_acceleration':True,
            'edge_subset_sz':1000000
        }, 
        
    }
    path_str=tacos.train(train_args,output_path)
    adata.obsm['integrated'] = tacos.embedding
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust)
    adata.obs['benchmark_cluster'] = adata.obs[clust]
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata