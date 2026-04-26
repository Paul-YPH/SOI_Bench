// Demo: process for computing evaluation metrics on method outputs.
// Metrics are computed conditionally based on what the method produced
// and what the dataset config specifies (alignment, clustering, multi-omics, etc.).
//
// Metric groups:
//   Spatial pattern   – Moran's I, CHAOS, PAS, ASW-spatial, SCS
//   Batch correction  – ASW-batch/annotation/F1, cLISI, iLISI, LISI-F1, BEMS
//   Clustering        – ARI, NMI, Homogeneity, Completeness, Purity, V-measure
//   Mapping           – CI, PCC (pair/grid), SSIM (pair/grid)
//   Alignment extras  – AAS (rotation), MAE (distortion)
//   Cross-slice       – PAA, LTARI, CLC
//
// Required params (set in nextflow.config):
//   params.metrics_py_dir – directory containing metrics_py module
//   params.shared_py_dir  – shared utilities (utils.py)
//   params.out_dir        – base output directory

process RunEvaluation {
    label 'python_cpu_process'

    tag "${dataset}_${tool}"

    container "/path/to/your/env.sif"
    publishDir path: "${params.out_dir}/${dataset}/${tool}", mode: 'copy'

    input:
    tuple val(dataset), path(h5ad), val(tool), val(py_dict_string)

    output:
    path("*.csv")

    script:
    """
    python -c "
    import sys
    import pandas as pd
    import json
    import os
    import scanpy as sc
    import numpy as np
    sys.path.append('${params.metrics_py_dir}')
    sys.path.append('${params.shared_py_dir}')
    from utils import get_ann_list, check_adata

    master_metrics_list = []

    meta_adata = sc.read_h5ad('${h5ad}')
    if meta_adata.uns['config']['multiomics_one_slice'] or meta_adata.uns['config']['multiomics_cross_slice']:
        adata = meta_adata.copy()
    else:
        adata = sc.read_h5ad(meta_adata.uns['config']['adata_path'])
        ann_list, _ = get_ann_list(adata)
        for a in ann_list:
            a.var_names_make_unique()
            a.obs_names_make_unique()
        adata = sc.concat(ann_list, join='inner', index_unique=None)
        sc.pp.filter_cells(adata, min_genes=1)
        sc.pp.filter_genes(adata, min_cells=1)
        common_cell = adata.obs_names.intersection(meta_adata.obs_names)
        adata = adata[common_cell].copy()
        meta_adata = meta_adata[common_cell].copy()
        if 'spatial_aligned' in meta_adata.obsm.keys():
            adata.obsm['spatial_aligned'] = meta_adata.obsm['spatial_aligned']
        if 'integrated' in meta_adata.obsm.keys():
            adata.obsm['integrated'] = meta_adata.obsm['integrated']
        adata.uns = meta_adata.uns
        new_columns = meta_adata.obs.columns.difference(adata.obs.columns)
        adata.obs = adata.obs.join(meta_adata.obs[new_columns])
        sc.pp.filter_cells(adata, min_genes=1)
        sc.pp.filter_genes(adata, min_cells=1)

    metrics_list = []
    adata = check_adata(adata, label_key='Ground Truth')

    if 'pi_list' in adata.uns.keys():
        from metrics_py import compute_paa, compute_ltari, compute_clc
        adata_light = sc.AnnData(
            obs=adata.obs.copy(),
            obsm={k: v.copy() for k, v in adata.obsm.items()},
            uns=adata.uns
        )
        metrics_list.extend([compute_paa(adata_light), compute_ltari(adata_light), compute_clc(adata_light)])
        del adata_light
        for key in ('pi_list', 'matching_cell_ids_list', 'pi_index_list'):
            adata.uns.pop(key, None)
        import gc; gc.collect()

    if adata.n_obs > 100000:
        sc.pp.subsample(adata, n_obs=100000, random_state=42)

    skip_spatial = '${tool}' in {'STAligner', 'INSPIRE', 'SPIRAL'} and np.isnan(adata.obsm.get('spatial_aligned', np.array([np.nan]))).any()
    if skip_spatial:
        del adata.obsm['spatial_aligned']

    from metrics_py import compute_moran_I, compute_pas, compute_scs, compute_chaos, compute_asw_spatial
    metrics_list.extend([compute_moran_I(adata), compute_chaos(adata), compute_pas(adata),
                         compute_asw_spatial(adata), compute_scs(adata)])

    if 'integrated' in adata.obsm.keys() and len(adata.obs['batch'].unique()) > 1:
        from metrics_py import compute_asw_batch, compute_asw_annotation, compute_asw_f1, compute_clisi, compute_ilisi, compute_lisi_f1, compute_bems
        asw_batch_df = compute_asw_batch(adata)
        asw_ann_df   = compute_asw_annotation(adata)
        ilisi_df     = compute_ilisi(adata, k0=30)
        clisi_df     = compute_clisi(adata, k0=30)
        metrics_list.extend([asw_batch_df, asw_ann_df, compute_asw_f1(asw_batch_df, asw_ann_df),
                              clisi_df, ilisi_df, compute_lisi_f1(clisi_df, ilisi_df),
                              compute_bems(adata)])

    if 'benchmark_cluster' in adata.obs.columns:
        from metrics_py import compute_ari, compute_nmi, compute_hom, compute_com, compute_purity, compute_vmeasure
        metrics_list.extend([compute_ari(adata), compute_nmi(adata), compute_hom(adata),
                              compute_com(adata), compute_purity(adata), compute_vmeasure(adata)])

    one_slice  = meta_adata.uns['config'].get('multiomics_one_slice', False)
    cross_slice = meta_adata.uns['config'].get('multiomics_cross_slice', False)
    if not one_slice and 'spatial_aligned' in adata.obsm.keys() and not skip_spatial:
        from metrics_py import compute_ci, compute_pcc, compute_ssim
        metrics_list.append(compute_ci(adata))
        if not cross_slice:
            metrics_list.extend([compute_pcc(adata), compute_ssim(adata)])
        metrics_list.extend([compute_pcc(adata, grid_num=10), compute_ssim(adata, grid_num=10)])

    if adata.uns['config']['rotation'] and 'spatial_aligned' in adata.obsm.keys():
        from metrics_py import compute_aas
        metrics_list.append(compute_aas(adata))

    if adata.uns['config']['distortion'] and 'spatial_aligned' in adata.obsm.keys():
        from metrics_py import compute_mae
        metrics_list.append(compute_mae(adata))

    df = pd.concat(metrics_list, axis=0, ignore_index=True)
    df['tool']    = '${tool}'
    df['dataset'] = '${dataset}'
    args_dict = { ${py_dict_string} }
    for key, value in args_dict.items():
        df[key] = value
    master_metrics_list.append(df)

    final_df = pd.concat(master_metrics_list, axis=0, ignore_index=True) if master_metrics_list \
               else pd.DataFrame(columns=['metric', 'value', 'tool', 'dataset', 'group', 'seed'])

    known_cols = ['metric', 'value', 'tool', 'dataset', 'group', 'seed']
    dynamic_cols = sorted(c for c in final_df.columns if c not in set(known_cols))
    final_df = final_df[[c for c in known_cols if c in final_df.columns] + dynamic_cols]
    final_df.to_csv(adata.uns['filename'] + '_evaluation.csv', index=False)
    "
    """
}
