// Demo: process for loading and preprocessing spatial omics datasets.
// Each dataset's h5ad is filtered, then genes are selected by mode:
//   'all'       – keep all genes
//   'hvg<N>'    – top-N highly variable genes
//   'svg<N>'    – top-N spatially variable genes
//
// Required params (set in nextflow.config):
//   params.shared_py_dir  – directory containing utils.py and gene_selection.py
//   params.out_dir        – base output directory

process LoadDatasets {
    label 'r_process'

    tag "${dataset}"

    container "/path/to/your/env.sif"

    input:
    tuple val(dataset), path(h5ad), val(gene_selection_mode)

    output:
    tuple val(dataset), path("loaded_adata_${gene_selection_mode}.h5ad"), val(gene_selection_mode)

    script:
    """
    python -c "
    import sys
    import os
    sys.path.append('${params.shared_py_dir}')
    from utils import check_adata
    from gene_selection import compute_svg, compute_hvg
    import scanpy as sc
    adata = sc.read_h5ad('${h5ad}')

    original_h5ad_path = os.path.realpath('${h5ad}')
    if adata.uns['config'].get('adata_path') != original_h5ad_path:
        adata.uns['config']['adata_path'] = original_h5ad_path

    gene_selection_mode = '${gene_selection_mode}'
    for sample in adata.uns['dataset']:
        tmp = adata.uns['dataset'][sample].copy()
        tmp = check_adata(tmp, label_key='Ground Truth')
        tmp.var_names_make_unique()
        sc.pp.filter_cells(tmp, min_genes=3)
        sc.pp.filter_genes(tmp, min_cells=3)

        mode = 'all'
        n_top_genes = 3000
        if gene_selection_mode == 'all':
            mode = 'all'
        else:
            if gene_selection_mode.startswith('hvg'):
                mode = 'hvg'
            elif gene_selection_mode.startswith('svg'):
                mode = 'svg'
            import re
            numbers = re.findall(r'\\d+', gene_selection_mode)
            if numbers:
                n_top_genes = int(numbers[0])

        tmp.obs['adata_path'] = adata.uns['config']['adata_path']
        if mode == 'hvg' and tmp.n_vars > n_top_genes:
            adata.uns['dataset'][sample] = compute_hvg(tmp, n_top_genes=n_top_genes)
        elif mode == 'svg' and tmp.n_vars > n_top_genes:
            adata.uns['dataset'][sample] = compute_svg(tmp, sample, n_top_genes=n_top_genes)
        elif mode == 'all':
            adata.uns['dataset'][sample] = tmp

    adata.write_h5ad('loaded_adata_${gene_selection_mode}.h5ad')
    "
    """
}
