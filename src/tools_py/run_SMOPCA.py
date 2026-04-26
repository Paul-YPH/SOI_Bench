import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import gc
from SMOPCA.model import SMOPCA
from utils import get_ann_list, create_lightweight_adata, set_seed
from clustering import clustering


def compute_sparkx_svg(adata, n_top=2000, num_cores=8):
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()
    pandas2ri.activate()

    SPARK = importr('SPARK')
    base = importr('base')

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    count_mat = X.T.astype(np.float64)  # genes x cells

    coords = np.array(adata.obsm['spatial'], dtype=np.float64)

    count_r = base.matrix(
        count_mat,
        nrow=count_mat.shape[0],
        ncol=count_mat.shape[1],
        dimnames=ro.r.list(
            ro.StrVector(adata.var_names.tolist()),
            ro.StrVector(adata.obs_names.tolist())
        )
    )
    coords_r = ro.r.matrix(coords, nrow=coords.shape[0], ncol=coords.shape[1])

    result = SPARK.sparkx(count_r, coords_r, numCores=num_cores, option='mixture')
    res_mtest = result.rx2('res_mtest')
    res_df = pandas2ri.rpy2py(res_mtest)

    rownames = pd.Index(list(res_mtest.rownames)).astype(str)

    if rownames.str.fullmatch(r"\d+").all():
        idx = rownames.astype(int) - 1 
        res_df.index = adata.var_names[idx]
    else:
        res_df.index = rownames

    res_df = res_df.sort_values('adjustedPval')
    top_genes = res_df.head(n_top)

    numpy2ri.deactivate()
    pandas2ri.deactivate()

    return top_genes


def run_SMOPCA(adata, **args_dict):
    seed = args_dict['seed']
    clust = args_dict['clust']
    set_seed(seed)

    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()

    if 'gam' in config['sample_list'].tolist():
        adata_omics1 = ann_list[config['sample_list'].tolist().index('rna')]
        adata_omics2 = ann_list[config['sample_list'].tolist().index('gam')]
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()
        rna_svg = compute_sparkx_svg(adata_omics1, n_top=2000)
        gam_svg = compute_sparkx_svg(adata_omics2, n_top=2000)
        adata_omics1 = adata_omics1[:, rna_svg.index]
        adata_omics2 = adata_omics2[:, gam_svg.index]
    elif 'adt' in config['sample_list'].tolist():
        adata_omics1 = ann_list[config['sample_list'].tolist().index('rna')]
        adata_omics2 = ann_list[config['sample_list'].tolist().index('adt')]
        adata_omics1.var_names_make_unique()
        rna_svg = compute_sparkx_svg(adata_omics1, n_top=2000)
        adata_omics1 = adata_omics1[:, rna_svg.index]
    del ann_list
    gc.collect()

    adata_omics1.obs_names = [n.split('_', 1)[1] for n in adata_omics1.obs_names]
    adata_omics2.obs_names = [n.split('_', 1)[1] for n in adata_omics2.obs_names]
    
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)

    sc.pp.normalize_total(adata_omics2, target_sum=1e4)
    sc.pp.log1p(adata_omics2)
    sc.pp.scale(adata_omics2)

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

    X1 = adata_omics1.X
    X2 = adata_omics2.X
    pos = np.array(adata_omics1.obsm['spatial'])

    smopca = SMOPCA(Y_list=[X1.T, X2.T], Z_dim=20, pos=pos, intercept=False, omics_weight=False)
    smopca.estimateParams(sigma_init_list=(1, 1), tol_sigma=2e-5, sigma_xtol_list=(1e-6, 1e-6), gamma_init=5, estimate_gamma=True)
    z = smopca.calculatePosterior()

    adata = adata_omics1.copy()
    # del adata_omics1, adata_omics2
    # gc.collect()

    z_clean = np.ascontiguousarray(z, dtype=np.float64)
    if z_clean.shape[0] != adata_omics1.n_obs:
        z_clean = z_clean.T
    adata = adata_omics1.copy()
    del adata_omics1, adata_omics2
    gc.collect()
    adata.obsm['integrated'] = z_clean

    # adata.obsm['integrated'] = z
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust,mclust_version = "2")
    adata.obs['benchmark_cluster'] = adata.obs[clust]

    adata = create_lightweight_adata(adata, config, args_dict=args_dict)

    return adata
