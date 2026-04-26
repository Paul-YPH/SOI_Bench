import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects import NULL
from utils import validate_adata, check_adata

def mclust_R_v1(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    if res is None or res == NULL:
        print('Mclust fitting failed. We change the modelNames to EEI.')
        res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, 'EEI')
    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def mclust_R_v2(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    import numpy as np
    from scipy.sparse import issparse
    import rpy2.robjects as robjects
    from rpy2.robjects import StrVector, FloatVector
    from rpy2.robjects.packages import importr

    np.random.seed(random_seed)
    try:
        importr("mclust")
    except Exception:
        robjects.r.library("mclust")
        
    robjects.r['set.seed'](random_seed)
    rmclust = robjects.r['Mclust']

    data = adata.obsm[used_obsm]
    if issparse(data):
        data = data.toarray()
    
    data = np.ascontiguousarray(data, dtype=np.float64)
    if data.shape[0] != adata.n_obs:
        data = data.T
        
    nrow, ncol = data.shape

    r_vec = FloatVector(data.flatten('C')) 
    r_mat = robjects.r.matrix(r_vec, nrow=nrow, ncol=ncol, byrow=True)
    r_names = StrVector([f"Dim{i+1}" for i in range(ncol)])

    robjects.globalenv['temp_mat'] = r_mat
    robjects.globalenv['temp_names'] = r_names
    robjects.r('temp_mat <- as.matrix(temp_mat)')
    robjects.r('colnames(temp_mat) <- temp_names')

    try:
        res = robjects.r(f'Mclust(temp_mat, G={int(num_cluster)}, modelNames="{modelNames}")')
    except Exception as e:
        print(f"Mclust ({modelNames}) error: {e}")
        try:
            res = robjects.r(f'Mclust(temp_mat, G={int(num_cluster)}, modelNames="EEI")')
        except Exception as e2:
            print(f"Mclust (EEI) error: {e2}")
            robjects.r('rm(temp_mat, temp_names)')
            raise

    try:
        mclust_res = np.array(res.rx2('classification'))
    except:
        mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res.astype(int).astype(str)
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    robjects.r('rm(temp_mat, temp_names)')
    
    return adata

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.0, end=3.0, max_steps=50):
    print(f'Searching resolution for {n_clusters} clusters using {method}...')
    
    if 'neighbors' not in adata.uns:
        print("Computing neighbors...")
        sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    
    def run_clustering(res):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            return len(adata.obs['leiden'].unique())
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            return len(adata.obs['louvain'].unique())
        else:
            raise ValueError("Method must be 'leiden' or 'louvain'")

    count_start = run_clustering(start)
    count_end = run_clustering(end)
    
    print(f"Checking boundaries: res={start}->{count_start} clusters, res={end}->{count_end} clusters")

    while count_start > n_clusters:
        end = start
        count_end = count_start
        start = start / 2.0
        if start < 1e-4:
            print("Resolution too close to 0, cannot find target clusters.")
            return start
        count_start = run_clustering(start)
        print(f"Expanding lower bound: res={start} -> {count_start} clusters")

    while count_end < n_clusters:
        start = end
        count_start = count_end
        end = end * 2.0
        count_end = run_clustering(end)
        print(f"Expanding upper bound: res={end} -> {count_end} clusters")

    history = {}
    
    lb = start
    ub = end
    
    for i in range(max_steps):
        res = (lb + ub) / 2
        count_unique = run_clustering(res)
        print(f'Step {i+1}: resolution={res:.4f}, cluster number={count_unique}')
        
        history[res] = count_unique
        
        if count_unique == n_clusters:
            print(f"Found exact match: resolution={res}")
            return res
        
        if count_unique > n_clusters:
            ub = res 
        else:
            lb = res 

    print(f"Exact match not found after {max_steps} steps.")
    closest_res = min(history.keys(), key=lambda x: abs(history[x] - n_clusters))
    closest_clusters = history[closest_res]
    
    print(f"Returning closest match: resolution={closest_res} with {closest_clusters} clusters.")
    
    return closest_res

 
def clustering(adata, 
               n_clusters=None, 
               use_rep='integrated', 
               label_key=None, 
               method='mclust', 
               start=0.0, 
               end=3.0,
               max_steps=50,
               mclust_version = "1"):
    
    print("Starting clustering...")
    validate_adata(adata, use_rep=use_rep, label_key=label_key)
    adata_tmp =check_adata(adata, use_rep=use_rep, label_key=label_key)
    
    if n_clusters is None and label_key is not None:
        n_clusters = adata_tmp.obs[label_key].nunique()
        print(f"Using {n_clusters} clusters from {label_key}")
    else:
        n_clusters = 10
        print("Using default 10 clusters")

    if method == 'mclust':
       print("Running Mclust clustering...")
       if mclust_version == "1":
            adata_tmp = mclust_R_v1(adata_tmp, used_obsm=use_rep, num_cluster=n_clusters)
            print("Mclust clustering completed")
       elif mclust_version == "2":
            adata_tmp = mclust_R_v2(adata_tmp, used_obsm=use_rep, num_cluster=n_clusters)
            print("Mclust clustering completed")
        
    elif method == 'leiden':
       print("Running Leiden clustering...")
       res = search_res(adata_tmp, n_clusters, use_rep=use_rep, method=method, start=start, end=end, max_steps=max_steps)
       sc.tl.leiden(adata_tmp, random_state=0, resolution=res)
       print("Leiden clustering completed")
       
    elif method == 'louvain':
       print("Running Louvain clustering...")
       res = search_res(adata_tmp, n_clusters, use_rep=use_rep, method=method, start=start, end=end, max_steps=max_steps)
       sc.tl.louvain(adata_tmp, random_state=0, resolution=res)
       print("Louvain clustering completed")

    return adata_tmp