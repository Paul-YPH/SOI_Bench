import os
import subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from rpy2.robjects import r, pandas2ri
import anndata2ri

def compute_svg(adata, sample_name, num_cores=20,n_top_genes=3000, r_script_path='/home/py95/project_pi_xz735/py95/SOI_bench/code/shared_py/run_sparkx.R'):
    if 'adata_path' not in adata.obs.columns:
        raise ValueError("adata.obs must contain 'adata_path' to determine output directory.")
        
    h5ad_file_path = adata.obs['adata_path'].values[0]
    path_spark = os.path.dirname(h5ad_file_path)
    rds_path = os.path.join(path_spark, f"{sample_name}_spark.rds")
    csv_path = os.path.join(path_spark, f"{sample_name}.csv")

    anndata2ri.activate()
    pandas2ri.activate()

    if sp.issparse(adata.X):
        sparse_counts = sp.csc_matrix(adata.X.T)
    else:
        sparse_counts = sp.csc_matrix(sp.csr_matrix(adata.X).T)
    
    if 'X' in adata.obs.columns and 'Y' in adata.obs.columns:
        location_df = adata.obs[['X', 'Y']]
    else:
        print("Using adata.obsm['spatial'] for coordinates...")
        location_df = pd.DataFrame(adata.obsm['spatial'][:, 0:2], columns=['X', 'Y'], index=adata.obs_names)

    r.assign("py_counts", sparse_counts)
    r.assign("py_location", location_df)
    r.assign("save_rds_path", rds_path)
    
    r.assign("py_spot_names", adata.obs_names.tolist()) 

    print("Generating SPARK object and saving RDS (in-process R)...")
    r("""
    library(SPARK)
    library(Matrix)

    colnames(py_counts) <- py_spot_names
    rownames(py_location) <- py_spot_names

    spark <- CreateSPARKObject(counts=py_counts, 
                            location=py_location,
                            percentage = 0, 
                            min_total_counts = 0)
                            
    spark@lib_size <- apply(spark@counts, 2, sum)

    saveRDS(spark, save_rds_path)
    """)

    if not os.path.exists(r_script_path):
        raise FileNotFoundError(f"Worker script not found at: {r_script_path}")

    cmd = [
        "Rscript",
        r_script_path,  
        rds_path,      
        csv_path,       
        str(num_cores),
        str(n_top_genes)
    ]

    print(f"Launching external R process for sparkx (Cores: {num_cores})...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Success! Results saved to: {csv_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running R script for {sample_name}")
        raise e
    
    result = pd.read_csv(csv_path, index_col=0)
    adata = adata[:,result.index].copy()
    return adata


def compute_hvg(adata, n_top_genes=3000):
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.shape[1]), flavor="seurat_v3")
    adata = adata[:,adata.var['highly_variable']].copy()
    return adata