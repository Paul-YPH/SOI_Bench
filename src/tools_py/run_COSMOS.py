import warnings
warnings.filterwarnings("ignore")

from COSMOS import cosmos
import scanpy as sc
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed
from clustering import clustering

def run_COSMOS(adata, **args_dict):
        seed = args_dict['seed']
        clust = args_dict['clust']
        knn = args_dict['knn']
        set_seed(seed)
        
        ann_list, config = get_ann_list(adata)
        del adata
        gc.collect()
        
        if 'atac' in config['sample_list'].tolist():
                adata_omics1 = ann_list[config['sample_list'].tolist().index('rna')]
                adata_omics2 = ann_list[config['sample_list'].tolist().index('atac')]       
        elif 'adt' in config['sample_list'].tolist():
                adata_omics1 = ann_list[config['sample_list'].tolist().index('rna')]
                adata_omics2 = ann_list[config['sample_list'].tolist().index('adt')]  
                sc.pp.normalize_total(adata_omics1, target_sum=1e4)
                sc.pp.log1p(adata_omics1)
                sc.pp.normalize_total(adata_omics2, target_sum=1e4)
                sc.pp.log1p(adata_omics2)
                
        del ann_list
        gc.collect()

        adata_omics1.obs_names = [n.split('_', 1)[1] for n in adata_omics1.obs_names]
        adata_omics2.obs_names = [n.split('_', 1)[1] for n in adata_omics2.obs_names]

        common_index = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
        if len(common_index) == 0:
            # obs_names have modality prefix (e.g. "rna_s1-BARCODE"); match by barcode suffix
            def strip_prefix(names):
                return [n.split('-', 1)[1] if '-' in n else n for n in names]
            bc1 = strip_prefix(adata_omics1.obs_names)
            bc2 = strip_prefix(adata_omics2.obs_names)
            common_bc = set(bc1) & set(bc2)
            idx1 = [i for i, b in enumerate(bc1) if b in common_bc]
            idx2 = [i for i, b in enumerate(bc2) if b in common_bc]
            # reorder idx2 to match the barcode order of idx1
            bc2_map = {b: i for i, b in enumerate(bc2)}
            idx2 = [bc2_map[bc1[i]] for i in idx1]
            adata_omics1 = adata_omics1[idx1].copy()
            adata_omics2 = adata_omics2[idx2].copy()
        else:
            adata_omics1 = adata_omics1[common_index].copy()
            adata_omics2 = adata_omics2[common_index].copy()

        adata_omics1.obs['x_pos'] = adata_omics1.obsm['spatial'][:,0]
        adata_omics1.obs['y_pos'] = adata_omics1.obsm['spatial'][:,1]
        adata_omics2.obs['x_pos'] = adata_omics2.obsm['spatial'][:,0]
        adata_omics2.obs['y_pos'] = adata_omics2.obsm['spatial'][:,1]

        ## COSMOS training 
        cosmos_comb = cosmos.Cosmos(adata1=adata_omics1,adata2=adata_omics2)
        cosmos_comb.preprocessing_data(n_neighbors = knn)
        cosmos_comb.train(spatial_regularization_strength=0.01, z_dim=50, 
                lr=1e-3, wnn_epoch = 500, total_epoch=1000, max_patience_bef=10, max_patience_aft=30, min_stop=200, 
                random_seed=seed, gpu=0, regularization_acceleration=True, edge_subset_sz=1000000)
        weights = cosmos_comb.weights

        adata = adata_omics1.copy()
        del adata_omics1, adata_omics2
        adata.obsm['integrated'] = cosmos_comb.embedding
        adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust,mclust_version = "2")
        adata.obs['benchmark_cluster'] = adata.obs[clust]    
           
        adata = create_lightweight_adata(adata, config, args_dict=args_dict)

        return adata
