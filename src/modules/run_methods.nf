// Demo: one process shown as a template (CPU/R-based tool).
// All other tools follow the same pattern — differ only in:
//   - container image
//   - label ('r_process' | 'python_cpu_process' | 'python_gpu_process')
//   - the imported run_<Tool> function and its call
//
// Required params (set in nextflow.config):
//   params.shared_py_dir  – shared Python utilities (utils.py, etc.)
//   params.tools_py_dir   – per-tool runner scripts (run_BANKSY.py, etc.)
//   params.out_dir        – base output directory
//
// Tools implemented (one process each):
//   BANKSY, CAST, CellCharter, COSMOS, DeepST, GPSA, GraphST, INSPIRE,
//   moscot (non-rigid / rigid), NicheCompass (batch / omics), PASTE, PASTE2,
//   PRECAST, SANTO, scSLAT, SEDR, SMOPCA, SPACEL, SpaDo, spaMGCN,
//   Spateo (rigid / non-rigid), SpatialGlue, spatiAlign, spaVAE (batch / omics),
//   spCLUE, SPIRAL, SSGATE, STADIA, STAGATE, STalign, STAligner, STAMP,
//   stClinic, STG3Net, STitch3D, stMSA, Tacos

process RunBANKSY {
    label 'r_process'

    tag "${dataset}_${tool}_seed${seed}"

    container "/path/to/your/env.sif"
    publishDir path: "${params.out_dir}/${dataset}/${tool}", mode: 'copy'

    input:
    tuple val(dataset), path(h5ad), val(tool), val(py_dict_string), val(seed)

    output:
    tuple val(dataset), path('*.h5ad'), val(tool), val(py_dict_string)

    script:
    """
    python -c "
    import sys
    import scanpy as sc
    import numpy as np

    sys.path.append('${params.tools_py_dir}')
    sys.path.append('${params.shared_py_dir}')

    from run_BANKSY import run_BANKSY

    args_dict = { ${py_dict_string} }
    adata = sc.read_h5ad('${h5ad}')

    adata.uns['dataset'] = {
        k: v for k, v in adata.uns['dataset'].items()
        if k != 'atac'
    }
    adata.uns['config']['sample_list'] = np.array(
        [s for s in adata.uns['config']['sample_list'] if s != 'atac']
    )

    adata = run_BANKSY(adata, **args_dict)

    adata.uns['config']['tool'] = '${tool}'
    adata.write_h5ad(adata.uns['filename'] + '.h5ad')
    "
    """
}

// All remaining tools follow the same structure as RunBANKSY above.
// GPU-based tools additionally wrap the call in a ResourceMonitor context
// to record peak GPU/CPU memory and wall-clock duration, e.g.:
//
//   from utils import ResourceMonitor
//   with ResourceMonitor(interval=1) as rm:
//       adata = run_<Tool>(adata, output_path=tmp_path, **args_dict)
//       ...
//   report = pd.DataFrame([{
//       'tag': '${dataset}_${tool}_seed${seed}',
//       'gpu_model': rm.gpu_model,
//       'peak_gpu_mb': rm.max_gpu_memory,
//       'peak_cpu_ram_mb': rm.max_cpu_mb,
//       'duration_min': rm.duration_min
//   }])
//   report.to_csv(...)
