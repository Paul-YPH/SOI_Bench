import warnings
warnings.filterwarnings("ignore")

import SSGATE as ssgate
import scanpy as sc
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed
from clustering import clustering

def run_SSGATE(adata, **args_dict):
    seed = args_dict['seed']
    clust = args_dict['clust']
    knn = args_dict['knn']
    set_seed(seed)
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    adata_omics1 = ann_list[config['sample_list'].tolist().index('rna')]
    adata_omics2 = ann_list[config['sample_list'].tolist().index('adt')]

    adata_omics1.obs_names = [n.split('_', 1)[1] for n in adata_omics1.obs_names]
    adata_omics2.obs_names = [n.split('_', 1)[1] for n in adata_omics2.obs_names]
    
    del ann_list
    gc.collect()
    
    common_index = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
    if len(common_index) == 0:
        def strip_prefix(names):
            return [n.split('-', 1)[1] if '-' in n else n for n in names]
        bc1 = strip_prefix(adata_omics1.obs_names)
        bc2 = strip_prefix(adata_omics2.obs_names)
        common_bc = set(bc1) & set(bc2)
        idx1 = [i for i, b in enumerate(bc1) if b in common_bc]
        bc2_map = {b: i for i, b in enumerate(bc2)}
        idx2 = [bc2_map[bc1[i]] for i in idx1]
        adata_omics1 = adata_omics1[adata_omics1.obs_names[idx1]].copy()
        adata_omics2 = adata_omics2[adata_omics2.obs_names[idx2]].copy()
    else:
        adata_omics1 = adata_omics1[common_index].copy()
        adata_omics2 = adata_omics2[common_index].copy()

    adata_omics1, adata_omics2 = ssgate.preprocess_cluster(adata_omics1, adata_omics2, res_st = 0.2, res_sp = 0.2, show_fig = False, figsize = (6,3))
    
    adata_omics1 = ssgate.Cal_Nbrs_Net(adata_omics1, feat = "X_pca", k_cutoff = knn, model = "KNN")
    adata_omics1 = ssgate.prune_net(adata_omics1)
    ssgate.Stats_Nbrs_Net(adata_omics1)

    sc.tl.pca(adata_omics2, svd_solver = 'arpack')
    adata_omics2 = ssgate.Cal_Nbrs_Net(adata_omics2, feat = "X_pca", k_cutoff = knn, model = "KNN")
    adata_omics2 = ssgate.prune_net(adata_omics2)
    ssgate.Stats_Nbrs_Net(adata_omics2)
    
    adata_omics1, adata_omics2 = ssgate.train(adata_omics1, adata_omics2, 
                                        hidden_dims1 = 128, 
                                        hidden_dims2 = 128, 
                                        out_dims = 30, 
                                        cluster_update_epoch = 50, 
                                        epochs_init = 50, 
                                        n_epochs=300, 
                                        save_reconstrction=False, 
                                        sigma = 0.1, 
                                        device = "cuda:0", 
                                        feat1 = "PCA",
                                        key_added = 'ssgate_embed')
    
    # Clustering
    print('### Clustering...')
    adata = adata_omics1.copy()
    
    del adata_omics1, adata_omics2
    gc.collect()
    
    adata.obsm['integrated'] = adata.obsm['ssgate_embed']
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust,mclust_version = "2")
    adata.obs['benchmark_cluster'] = adata.obs[clust]
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata