from .aas import compute_aas
from .paa import compute_paa
from .ari import compute_ari
from .asw_spatial import compute_asw_spatial
from .asw import compute_asw_celltype, compute_asw_batch, compute_asw_f1
from .bems import compute_bems
from .chaos import compute_chaos
from .ci import compute_ci
from .clc import compute_clc
from .com import compute_com
from .hom import compute_hom
from .lisi import compute_ilisi, compute_clisi, compute_lisi_f1
from .ltari import compute_ltari
from .nmi import compute_nmi
from .pas import compute_pas
from .pcc import compute_pcc
from .scs import compute_scs
from .ssim import compute_ssim
from .mae import compute_mae
from .utils import *
import pandas as pd
import time

def evaluation(adata, 
               metrics_list = None, 
               cluster_option = None,
               spatial_key = 'spatial',
               spatial_aligned_key = 'spatial_aligned',
               label_key = 'Ground Truth',
               use_rep = 'integrated',
               batch_key = 'batch',
               matching_list = None,
               matching_ground_truth_list = None,
               pi_list = None,
               used_time = None,
               used_cpu_memory = None,
               used_gpu_memory = None,
               used_memory = None,
               angle_true = None,
               peak_data = False):
    
    print("Starting evaluation...")
    if not peak_data:
        sc.pp.filter_genes(adata, min_cells=1)
        adata = process_anndata(adata, highly_variable_genes=True, normalize_total=True, log1p=True)

    if cluster_option in adata.obs.columns:
        print("Using benchmark cluster..."+cluster_option)
        adata.obs['benchmark_cluster'] = adata.obs[cluster_option]
        cluster_option = 'benchmark_cluster'
    
    if 'embedding' in metrics_list:
        validate_adata(adata, use_rep=use_rep, label_key=label_key)
        adata_tmp = check_adata(adata, use_rep=use_rep, label_key=label_key)
        
        print("- Computing embedding metrics...")
        start = time.time()
        ilisi_df = compute_ilisi(adata = adata_tmp, batch_key=batch_key, use_rep=use_rep,k0=30)###
        end = time.time()
        print(f"Time taken for ilisi: {end - start} seconds")
        start = time.time()
        clisi_df = compute_clisi(adata = adata_tmp, label_key=label_key, use_rep=use_rep,k0=30)###
        end = time.time()
        print(f"Time taken for clisi: {end - start} seconds")
        lisi_f1_df = compute_lisi_f1(ilisi_df, clisi_df)

        print("- Computing ASW metrics...")
        start = time.time()
        asw_celltype_df = compute_asw_celltype(adata = adata_tmp, label_key=label_key, use_rep=use_rep)
        end = time.time()
        print(f"Time taken for asw_celltype: {end - start} seconds")
        start = time.time()
        asw_batch_df = compute_asw_batch(adata = adata_tmp, label_key=label_key, use_rep=use_rep, batch_key=batch_key)
        end = time.time()
        print(f"Time taken for asw_batch: {end - start} seconds")
        asw_f1_df = compute_asw_f1(asw_batch_df, asw_celltype_df)

        print("- Computing BEMS...")
        start = time.time()
        bems_df = compute_bems(adata_tmp, use_rep=use_rep, batch_key=batch_key)
        end = time.time()
        print(f"Time taken for bems: {end - start} seconds")

    if 'clustering' in metrics_list:
        print("Computing clustering metrics...")
        start = time.time()
        ari_df = compute_ari(adata = adata_tmp, label_key=label_key, cluster_key=cluster_option)
        end = time.time()
        print(f"Time taken for ari: {end - start} seconds")
        start = time.time()
        nmi_df = compute_nmi(adata = adata_tmp, label_key=label_key, cluster_key=cluster_option)
        end = time.time()
        print(f"Time taken for nmi: {end - start} seconds")
        start = time.time()
        hom_df = compute_hom(adata = adata_tmp, label_key=label_key, cluster_key=cluster_option)
        end = time.time()
        print(f"Time taken for hom: {end - start} seconds")
        start = time.time()
        com_df = compute_com(adata = adata_tmp, label_key=label_key, cluster_key=cluster_option)
        end = time.time()
        print(f"Time taken for com: {end - start} seconds")
        
    if 'pattern' in metrics_list and 'embedding' in metrics_list:
        validate_adata(adata, use_rep=use_rep, label_key=label_key, spatial_key=spatial_key)
        adata_tmp = check_adata(adata, use_rep=use_rep, label_key=label_key, spatial_key=spatial_key)
        print("Computing pattern metrics with embedding...")
        
        start = time.time()
        chaos_df = compute_chaos(adata = adata_tmp, cluster_key=cluster_option, batch_key=batch_key)
        end = time.time()
        print(f"Time taken for chaos: {end - start} seconds")
        start = time.time()
        pas_df = compute_pas(adata = adata_tmp, cluster_key=cluster_option, batch_key=batch_key)
        end = time.time()
        print(f"Time taken for pas: {end - start} seconds")
        start = time.time()
        asw_spatial_df = compute_asw_spatial(adata = adata_tmp, cluster_key=cluster_option, batch_key=batch_key)
        end = time.time()
        print(f"Time taken for asw_spatial: {end - start} seconds")
        start = time.time()
        scs_df = compute_scs(adata = adata_tmp, cluster_key=cluster_option, batch_key=batch_key)
        end = time.time()
        print(f"Time taken for scs: {end - start} seconds")

    elif 'pattern' in metrics_list and 'mapping' in metrics_list:
        validate_adata(adata, label_key=label_key, spatial_key=spatial_aligned_key)
        adata_tmp = check_adata(adata, label_key=label_key, spatial_key=spatial_aligned_key)
        print("Computing pattern metrics with mapping...")
        
        start = time.time()
        chaos_df = compute_chaos(adata = adata_tmp, label_key=label_key, spatial_key=spatial_aligned_key)
        end = time.time()
        print(f"Time taken for chaos: {end - start} seconds")
        start = time.time()
        pas_df = compute_pas(adata = adata_tmp, label_key=label_key, spatial_key=spatial_aligned_key)
        end = time.time()
        print(f"Time taken for pas: {end - start} seconds")
        start = time.time()
        asw_spatial_df = compute_asw_spatial(adata = adata_tmp, label_key=label_key, spatial_key=spatial_aligned_key)
        end = time.time()
        print(f"Time taken for asw_spatial: {end - start} seconds")
        start = time.time()
        scs_df = compute_scs(adata = adata_tmp, label_key=label_key, spatial_key=spatial_aligned_key)
        end = time.time()
        print(f"Time taken for scs: {end - start} seconds")
        
    if 'mapping' in metrics_list:
        validate_adata(adata, label_key=label_key, spatial_key=spatial_aligned_key)
        adata_tmp = check_adata(adata, label_key=label_key, spatial_key=spatial_aligned_key)
        print("Computing mapping metrics...")
        
        if not peak_data:
            start = time.time()
            pcc_prop_df = compute_pcc(adata = adata_tmp, grid_num=10, spatial_key=spatial_aligned_key)
            end = time.time()
            print(f"Time taken for pcc_prop: {end - start} seconds")
            start = time.time()
            pcc_exp_df = compute_pcc(adata = adata_tmp, spatial_key=spatial_aligned_key)
            end = time.time()
            print(f"Time taken for pcc_exp: {end - start} seconds")
            start = time.time()
            ssim_prop_df = compute_ssim(adata = adata_tmp, grid_num=10, spatial_key=spatial_aligned_key)
            end = time.time()
            print(f"Time taken for ssim_prop: {end - start} seconds")
            start = time.time()
            ssim_exp_df = compute_ssim(adata = adata_tmp, spatial_key=spatial_aligned_key)
            end = time.time()
            print(f"Time taken for ssim_exp: {end - start} seconds")
        else:
            start = time.time()
            pcc_prop_df = compute_pcc(adata = adata_tmp, grid_num=10, spatial_key=spatial_aligned_key)
            end = time.time()
            print(f"Time taken for pcc_prop: {end - start} seconds")
            start = time.time()
            ssim_prop_df = compute_ssim(adata = adata_tmp, grid_num=10, spatial_key=spatial_aligned_key)
            end = time.time()
            print(f"Time taken for ssim_prop: {end - start} seconds")

            pcc_exp_df = pd.DataFrame({'metric': ['pcc_pair'], 'value': [pd.NA], 'group': [pd.NA]})
            ssim_exp_df = pd.DataFrame({'metric': ['ssim_pair'], 'value': [pd.NA], 'group': [pd.NA]})
            
        start = time.time()
        ci_df = compute_ci(adata = adata_tmp,spatial_key=spatial_aligned_key)
        end = time.time()
        print(f"Time taken for ci: {end - start} seconds")
        
    if 'matching' in metrics_list:
        
        min_length = 10
        all_valid = all(len(sublist) > min_length for sublist in matching_list)

        if all_valid:
            start = time.time()
            clc_df = compute_clc(adata=adata_tmp, label_key=label_key, pi=pi_list, spatial_key=spatial_aligned_key)
            end = time.time()
            print(f"Time taken for clc: {end - start} seconds")
            start = time.time()
            paa_df = compute_paa(adata=adata_tmp, label_key=label_key, matching_list=matching_list)
            end = time.time()
            print(f"Time taken for paa: {end - start} seconds")
            start = time.time()
            ltari_df = compute_ltari(adata=adata_tmp, label_key=label_key, matching_list=matching_list)
            end = time.time()
            print(f"Time taken for ltari: {end - start} seconds")
        else:
            print(f"Warning: Some sublists in matching_list have length <= {min_length}. Setting all metrics to NA.")
            clc_df = pd.DataFrame({'metric': ['clc'], 'value': [pd.NA], 'group': [pd.NA]})
            paa_df = pd.DataFrame({'metric': ['paa'], 'value': [pd.NA], 'group': [pd.NA]})
            ltari_df = pd.DataFrame({'metric': ['ltari'], 'value': [pd.NA], 'group': [pd.NA]})
        
    if angle_true is not None and 'r_transform' in metrics_list:
        print("Computing r_transform metrics...")
        start = time.time()
        aas_df = compute_aas(adata = adata_tmp, angle_true=angle_true, matching_ground_truth=matching_ground_truth_list)
        end = time.time()
        print(f"Time taken for aas: {end - start} seconds")

    if 'nr_transform' in metrics_list and 'spatial_original' in adata.obsm:
        print("Computing nr_transform metrics...")
        start = time.time()
        mae_df = compute_mae(adata = adata_tmp, spatial_aligned_key='spatial_aligned', spatial_original_key='spatial_original')
        end = time.time()
        print(f"Time taken for mae: {end - start} seconds")
        
    print("Compiling results...")

    df_list = []
    
    if 'embedding' in metrics_list:
        df_list.extend([ilisi_df, clisi_df, lisi_f1_df, asw_celltype_df, asw_batch_df, asw_f1_df, bems_df])

    if 'clustering' in metrics_list:
        df_list.extend([ari_df, nmi_df, hom_df, com_df])

    if 'pattern' in metrics_list:
        df_list.extend([chaos_df, pas_df, asw_spatial_df, scs_df])

    if 'mapping' in metrics_list:
        df_list.extend([pcc_prop_df, pcc_exp_df, ssim_prop_df, ssim_exp_df, ci_df])

    if 'matching' in metrics_list:
        df_list.extend([paa_df, ltari_df, clc_df])

    if 'r_transform' in metrics_list and angle_true is not None:
        df_list.extend([aas_df])

    if 'nr_transform' in metrics_list and 'spatial_original' in adata.obsm:
        df_list.extend([mae_df])
        
    metrics_df = pd.concat(df_list, ignore_index=True)
    runtime_df = pd.DataFrame({'metric': ['ct'], 'value': [used_time], 'group': ['runtime']})
    memory_df = pd.DataFrame({'metric': ['mu'], 'value': [used_memory], 'group': ['memory']})
    cpu_memory_df = pd.DataFrame({'metric': ['cpu_memory'], 'value': [used_cpu_memory], 'group': ['memory']})
    gpu_memory_df = pd.DataFrame({'metric': ['gpu_memory'], 'value': [used_gpu_memory], 'group': ['memory']})
    metrics_df = pd.concat([metrics_df, runtime_df, memory_df, cpu_memory_df, gpu_memory_df], ignore_index=True)

    print("Evaluation completed!")

    return metrics_df