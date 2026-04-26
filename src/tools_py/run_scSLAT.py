import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import numpy as np
import pandas as pd
import cv2
from itertools import chain
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
import gc
from scSLAT.model import Cal_Spatial_Net, load_anndatas, run_SLAT, spatial_match, run_SLAT_multi
from scSLAT.model.prematch import icp
from scSLAT.model.prematch.utils import alpha_shape
from utils import get_ann_list, create_lightweight_adata,set_seed

def calculate_alpha(points: np.ndarray, method: str = "mean_distance", factor: float = 2) -> float:
    nbrs = NearestNeighbors(n_neighbors=2).fit(points)
    distances, _ = nbrs.kneighbors(points)

    if method == "mean_distance":
        alpha = np.mean(distances[:, 1]) * factor
    elif method == "median_distance":
        alpha = np.median(distances[:, 1]) * factor
    elif method == "std_distance":
        alpha = np.std(distances[:, 1]) * factor
    else:
        raise ValueError("Invalid method. Choose 'mean_distance', 'median_distance', or 'std_distance'.")
    return alpha

def prematch(adata1, adata2):
    alpha1 = calculate_alpha(adata1.obsm['spatial_aligned'])
    boundary_1, edges_1, _ = alpha_shape(adata1.obsm['spatial_aligned'],alpha=alpha1, only_outer=True)
    alpha2 = calculate_alpha(adata2.obsm['spatial_aligned'])
    boundary_2, edges_2, _ = alpha_shape(adata2.obsm['spatial_aligned'],alpha=alpha2, only_outer=True)
    
    T, error = icp(adata2.obsm['spatial_aligned'][boundary_2,:].T, adata1.obsm['spatial_aligned'][boundary_1,:].T)
    rotation = np.arcsin(T[0,1]) * 360 / 2 / np.pi

    trans = np.squeeze(cv2.transform(np.array([adata2.obsm['spatial_aligned']], copy=True).astype(np.float32), T))[:,:2]
    adata2.obsm['spatial_aligned'] = trans
    adata2.obs['angle'] = rotation
    return adata2

def run_scSLAT(adata, **args_dict):
    seed = args_dict['seed']
    knn = args_dict['knn']
    pcs = args_dict['pcs']
    set_seed(seed)
    
    ann_list, config = get_ann_list(adata)
    if 'atac' in config['sample_list']:
        peak_data = True
    else:
        peak_data = False
    del adata
    gc.collect()
    
    for ann in ann_list:
        Cal_Spatial_Net(ann, k_cutoff=knn, model='KNN')
    
    #### Stack slices
    for i in range(len(ann_list)):
        ann_list[i].obsm['spatial_aligned'] = ann_list[i].obsm['spatial']
    for i in range(len(ann_list)-1):
        ann_list[i+1] = prematch(ann_list[i], ann_list[i+1])
    ### Perform SLAT
    if len(ann_list) == 2:
        if peak_data:
            import scglue
            import itertools
            
            for ann in ann_list:
                tech_type = ann.obs['batch'].iloc[0].lower() 
                if 'rna' in tech_type:
                    rna = ann
                elif 'atac' in tech_type:
                    atac = ann

            def format_var_name(var_name):
                parts = var_name.split("-")
                if len(parts) == 3:
                    return f"{parts[0]}:{parts[1]}-{parts[2]}"
                return var_name  

            atac.var.index = atac.var.index.to_series().apply(format_var_name)
            atac.var.names = atac.var.index
            scglue.data.lsi(atac, n_components=100, n_iter=15)
            sc.pp.neighbors(atac, use_rep="X_lsi", metric="cosine")
            
            rna.layers["counts"] = rna.X.copy()
            sc.pp.highly_variable_genes(rna, n_top_genes=2500, flavor="seurat_v3")
            sc.pp.normalize_total(rna)
            sc.pp.log1p(rna)
            sc.pp.scale(rna)
            sc.tl.pca(rna, n_comps=100, svd_solver="auto")
            sc.pp.neighbors(rna, metric="cosine")
            
            if rna.obs['species'].unique()[0] == 'mouse':
                gtf = "/nfs/roberts/home/py95/project_pi_xz735/py95/SOI_bench/code/tools_py/GLUE/gencode.vM30.annotation.gtf.gz"
            
            scglue.data.get_gene_annotation(
                rna, gtf=gtf,
                gtf_by="gene_name"
            )
            
            split = atac.var_names.str.split(r"[:-]")
            atac.var["chrom"] = split.map(lambda x: x[0] if len(x) > 0 else np.nan)
            atac.var["chromStart"] = split.map(lambda x: x[1] if len(x) > 1 else np.nan)
            atac.var["chromEnd"] = split.map(lambda x: x[2] if len(x) > 2 else np.nan)
            
            if rna.var["chromStart"].isna().any():
                rna = rna[:, rna.var["chromStart"].notna()].copy()

            if atac.var["chromStart"].isna().any():
                atac = atac[:, atac.var["chromStart"].notna()].copy()

            rna.var["chromStart"] = rna.var["chromStart"].astype(int)
            rna.var["chromEnd"] = rna.var["chromEnd"].astype(int)
            atac.var["chromStart"] = atac.var["chromStart"].astype(int)
            atac.var["chromEnd"] = atac.var["chromEnd"].astype(int)
            
            guidance = scglue.genomics.rna_anchored_prior_graph(rna, atac)
            guidance.number_of_nodes(), guidance.number_of_edges()
            scglue.models.configure_dataset(
                rna, "NB", use_highly_variable=True,
                use_layer="counts", use_rep="X_pca"
            )
            scglue.models.configure_dataset(
                atac, "NB", use_highly_variable=True,
                use_rep="X_lsi"
            )
            guidance = guidance.subgraph(itertools.chain(
                rna.var.query("highly_variable").index,
                atac.var.query("highly_variable").index
            ))

            glue = scglue.models.fit_SCGLUE(
                {"rna": rna, "atac": atac}, guidance,
                fit_kws={"directory": "glue"}
            )
            rna.obsm["X_glue"] = glue.encode_data("rna", rna)
            atac.obsm["X_glue"] = glue.encode_data("atac", atac)
            rna.obs['domain'] = 'rna'
            atac.obs['domain'] = 'atac'
            ann_list = [rna, atac]
            edges, features = load_anndatas([ann_list[0], ann_list[1]], feature='glue',dim = min(pcs, ann_list[0].shape[1], ann_list[0].shape[0]))
        else:
            edges, features = load_anndatas([ann_list[0], ann_list[1]], feature='DPCA',check_order=False,dim = min(pcs, ann_list[0].shape[1]-1, ann_list[0].shape[0]-1))
        embd0, embd1, time_tmp = run_SLAT(features, edges)
        best, index, distance = spatial_match([embd0, embd1], adatas=[ann_list[0],ann_list[1]], reorder=False)
        
        cell_ids_1 = ann_list[0].obs_names
        cell_ids_2 = ann_list[1].obs_names
        matching = np.array([range(index.shape[0]), best])
        matching_cell_ids = pd.DataFrame({
            'id_1': cell_ids_1[matching[1]],
            'id_2': cell_ids_2[matching[0]]     
        })
        rows = matching_cell_ids['id_1'].unique()
        cols = matching_cell_ids['id_2'].unique()
        results = pd.DataFrame(0, index=cell_ids_1, columns=cell_ids_2)
        for _, row in matching_cell_ids.iterrows():
            results.loc[row['id_1'], row['id_2']] = 1

        pi_list = [results.to_numpy()]
        pi_index_list = [cell_ids_1]
        pi_column_list = [cell_ids_2]
        matching_cell_ids_list = [matching_cell_ids]
    else:
        matching_list, zip_res = run_SLAT_multi(ann_list, k_cutoff=knn)
        
        pi_list = []
        pi_index_list = []
        pi_column_list = []
        matching_cell_ids_list = []
        for i in range(len(matching_list)):
            cell_ids_1 = ann_list[i].obs_names
            cell_ids_2 = ann_list[i+1].obs_names
            matching_cell_ids = pd.DataFrame({
                'id_1': cell_ids_1[matching_list[i][1]],
                'id_2': cell_ids_2[matching_list[i][0]]     
            })
            results = pd.DataFrame(0, index=cell_ids_1, columns=cell_ids_2)
            for _, row in matching_cell_ids.iterrows():
                results.loc[row['id_1'], row['id_2']] = 1
                
            pi_list.append(results.to_numpy())
            pi_index_list.append(cell_ids_1)
            pi_column_list.append(cell_ids_2)
            matching_cell_ids_list.append(matching_cell_ids)   
            
    adata = sc.concat(ann_list, index_unique=None)
    del ann_list
    gc.collect()
    
    adata = create_lightweight_adata(adata, config=config, pi_list=pi_list, pi_index_list=pi_index_list, pi_column_list=pi_column_list, matching_cell_ids_list=matching_cell_ids_list, args_dict=args_dict)
    
    return adata