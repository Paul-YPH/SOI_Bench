import warnings
warnings.filterwarnings("ignore")

import argparse
import time
import resource
import os
import sys
import scanpy as sc
import subprocess
import pynvml
import threading

max_cpu_memory = 0
max_gpu_memory = 0
stop_event = threading.Event()
pid = os.getpid()

current_dir = os.path.dirname(os.path.abspath(__file__)) 
benchmark_dir = os.path.dirname(current_dir)  
sys.path.append(benchmark_dir)
from metrics.evaluation import evaluation
from metrics.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type = str, default = None)
parser.add_argument('--output_path', type = str, default = None)
parser.add_argument('--sample', type = str, nargs="+", default = None)
parser.add_argument('--cluster_option', type = str, default = 'mclust')
parser.add_argument('--metrics', type = str, nargs="+", default = None)
parser.add_argument('--angle_true', type = int, nargs="+", default = None)
parser.add_argument('--rigid', type=lambda x: x.lower() in ['true', '1', 'yes'], default=True)
parser.add_argument('--overlap', type = float, nargs="+", default = None)
parser.add_argument('--multi_slice', type=lambda x: x.lower() in ['true', '1', 'yes'], default=False)
parser.add_argument('--pseudocount', type = int, default = None)
parser.add_argument('--rep', type = int, default = 1)
parser.add_argument('--tool_name', type = str, default = None)
parser.add_argument('--distortion', type = float, nargs="+", default = None)
parser.add_argument('--subsample', type=lambda x: x.lower() in ['true', '1', 'yes'], default=False)

args = parser.parse_args()
input_path = args.input_path
base_output_path = args.output_path
sample_list = args.sample
cluster_option = args.cluster_option
metrics_list = args.metrics
rigid_option = args.rigid
overlap_list = args.overlap
angle_true = args.angle_true
multi_slice = args.multi_slice
pseudocount = args.pseudocount
rep = args.rep
tool_name = args.tool_name
distortion_list = args.distortion
subsample = args.subsample

if not os.path.exists(base_output_path):
    print(f"Output path {base_output_path} does not exist, creating it...")
    os.makedirs(base_output_path, exist_ok=True)
    
if rigid_option is False and 'mapping' in metrics_list:
    print('### The tool used Non-rigid alignment!')
elif rigid_option is True and 'mapping' in metrics_list:
    print('### The tool used Rigid alignment!')
    
for sample in sample_list:
    if 'atac' in sample and tool_name == 'SLAT':
        sample = sample + '_peak'
        peak_data = True
    else:
        peak_data = False

print('### The input path is:', input_path)
print('### The output path is:', base_output_path)
print('### The sample list is:', sample_list)

def get_cpu_memory():
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage_kb / 1024

def get_gpu_memory_by_pid(target_pid):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)
    
    gpu_memory_used = 0
    for line in result.stdout.strip().split("\n"):
        try:
            process_pid, memory = map(int, line.split(", "))
            if process_pid == target_pid:
                gpu_memory_used += memory  
        except ValueError:
            continue 

    return gpu_memory_used  

def monitor_memory():
    global max_cpu_memory, max_gpu_memory  
    while not stop_event.is_set():  
        cpu_memory = get_cpu_memory()
        gpu_memory = get_gpu_memory_by_pid(pid)

        max_cpu_memory = max(max_cpu_memory, cpu_memory)
        max_gpu_memory = max(max_gpu_memory, gpu_memory)

        time.sleep(1) 
            
def run_benchmarking(input_path,
                     base_output_path,
                     sample_list,
                     cluster_option,
                     metrics_list,
                     rigid_option,
                     angle_true,
                     overlap_list,
                     distortion_list
                    #  pseudocount,
                    #  rep_num
                     ):
    #################### Paths ####################
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist")
    # output_path = generate_output_path(base_output_path, angle_true, overlap_list, pseudocount, distortion_list,rep_num)
    output_path = base_output_path
    if not os.path.exists(output_path):
        print(f"Output path {output_path} does not exist, creating it...")
        os.makedirs(output_path, exist_ok=True)

    ### initialize
    matching_ground_truth_list = None
    pi_list = None
    matching_cell_ids_list = None
    ann_list = []
    
    #################### Read Data ####################
    sample_list_path = []
    for sample in sample_list:
        adata_tmp = sc.read_h5ad(os.path.join(input_path, sample, sample+'.h5ad'))
        adata_tmp = check_adata(adata_tmp,label_key='Ground Truth')
        adata_tmp.X = adata_tmp.X.astype(np.float32)
        sample_list_path.append(os.path.join(input_path, sample, sample+'_spatialign.h5ad'))
                                
        ### spatial type
        spatial = adata_tmp.obsm["spatial"]
        if not isinstance(spatial, np.ndarray):
            spatial = spatial.to_numpy()
        if spatial.dtype != np.float32:
            spatial = spatial.astype(np.float32)
        adata_tmp.obsm["spatial"] = spatial
            
        ### array col/row
        if 'array_col' not in adata_tmp.obs.columns:
            adata_tmp.obs['array_col'] = adata_tmp.obsm['spatial'][:, 1]
        else:
            adata_tmp.obs['array_col'] = adata_tmp.obs['array_col'].astype(np.float32)
        if 'array_row' not in adata_tmp.obs.columns:
            adata_tmp.obs['array_row'] = adata_tmp.obsm['spatial'][:, 0] 
        else:
            adata_tmp.obs['array_row'] = adata_tmp.obs['array_row'].astype(np.float32)
        
        adata_tmp.obs['batch'] = sample
        adata_tmp.var_names_make_unique()
        sc.pp.filter_cells(adata_tmp, min_genes=1)
        sc.pp.filter_genes(adata_tmp, min_cells=1)
        print("Input data shape  ",sample, adata_tmp.shape)
        adata_tmp.obs['cell_index'] = adata_tmp.obs_names
        ann_list.append(adata_tmp)
        
    if os.path.exists(os.path.join(input_path, 'matching_ground_truth_list.npy')):
        tmp = np.load(os.path.join(input_path, 'matching_ground_truth_list.npy'), allow_pickle=True).item()
        matching_ground_truth_list = [tmp[str(i)] for i in range(len(tmp))]

    #################### Run Methods ####################
    pynvml.nvmlInit()

    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()

    start = time.time()
    python_path = os.path.abspath(sys.executable)
    env_name = python_path.split(os.sep)[python_path.split(os.sep).index('envs') + 1]

    if env_name == 'spateo' and rigid_option is False:
        print('### Using Spateo environment (non-rigid)')
        from run_Spateo_NR import run_Spateo_NR
        #ann_list = subsample_adata(ann_list, subsample, 10000)
        adata, pi_list, matching_cell_ids_list = run_Spateo_NR(ann_list)

    elif env_name == 'spateo' and rigid_option is True:
        print('### Using Spateo environment (rigid)')
        from run_Spateo_R import run_Spateo_R
        adata, pi_list, matching_cell_ids_list = run_Spateo_R(ann_list)

    elif env_name == 'cast':
        print('### Using CAST environment')
        from run_CAST import run_CAST
        ann_list = subsample_adata(ann_list, subsample, 20000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata = run_CAST(ann_list, sample_list, output_path)

    elif env_name == 'precast':
        print('### Using PRECAST environment')
        from run_PRECAST import run_PRECAST
        ann_list = subsample_adata(ann_list, subsample, 20000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata = run_PRECAST(ann_list)
        cluster_option = 'precast_cluster'

    elif env_name == 'paste':
        print('### Using PASTE environment')
        from run_PASTE import run_PASTE
        ann_list = subsample_adata(ann_list, subsample, 10000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata, pi_list, matching_cell_ids_list = run_PASTE(ann_list, sample_list)

    elif env_name == 'paste2':
        print('### Using PASTE2 environment')
        from run_PASTE2 import run_PASTE2
        ann_list = subsample_adata(ann_list, subsample, 5000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata, pi_list, matching_cell_ids_list = run_PASTE2(ann_list, sample_list,overlap_list)

    elif env_name == 'deepst':
        print('### Using DeepST environment')
        from run_DeepST import run_DeepST
        ann_list = subsample_adata(ann_list, subsample, 10000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        ann_list = filter_common_genes(ann_list)
        adata = run_DeepST(ann_list, sample_list, output_path)
        cluster_option = 'DeepST_refine_domain'

    elif env_name == 'gpsa':
        print('### Using GPSA environment')
        from run_GPSA import run_GPSA
        ann_list = subsample_adata(ann_list, subsample, 5000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        ann_list = filter_common_genes(ann_list)
        adata = run_GPSA(ann_list)

    elif env_name == 'graphst':
        print('### Using GraphST environment')
        from run_GraphST import run_GraphST
        ann_list = subsample_adata(ann_list, subsample, 20000) 
        adata = run_GraphST(ann_list, cluster_option)

    elif env_name == 'inspire':
        print('### Using INSPIRE environment')
        from run_INSPIRE import run_INSPIRE
        ann_list = subsample_adata(ann_list, subsample, 10000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata = run_INSPIRE(ann_list, cluster_option) 

    elif env_name == 'santo':
        print('### Using SANTO environment')
        from run_SANTO import run_SANTO
        ann_list = subsample_adata(ann_list, subsample, 10000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata = run_SANTO(ann_list,overlap_list)

    elif env_name == 'stagate':
        print('### Using STAGATE environment')
        from run_STAGATE import run_STAGATE
        adata = run_STAGATE(ann_list, cluster_option)

    elif env_name == 'spatialign':
        print('### Using spatiAlign environment')
        from run_spatiAlign import run_spatiAlign
        ann_list = subsample_adata(ann_list, subsample, 10000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        for ann_i in range(len(ann_list)):
            ann_list[ann_i].write(sample_list_path[ann_i])  
        adata = run_spatiAlign(ann_list, sample_list, output_path, cluster_option,sample_list_path)
        for file_path in sample_list_path:
            if os.path.exists(file_path) and file_path.endswith('.h5ad'):
                os.remove(file_path)

    elif env_name == 'stalign':
        print('### Using STalign environment')
        from run_STalign import run_STalign
        ann_list = subsample_adata(ann_list, subsample, 10000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata = run_STalign(ann_list)

    elif env_name == 'staligner':
        print('### Using STAligner environment')
        from run_STAligner import run_STAligner
        ann_list = subsample_adata(ann_list, subsample, 20000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata = run_STAligner(ann_list, sample_list, cluster_option)

    elif env_name == 'stg3net':
        print('### Using STG3Net environment')
        from run_STG3Net import run_STG3Net
        ann_list = subsample_adata(ann_list, subsample, 20000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata = run_STG3Net(ann_list, sample_list, cluster_option)

    elif env_name == 'stitch3d':
        print('### Using STitch3D environment')
        from run_STitch3D import run_STitch3D
        ann_list = subsample_adata(ann_list, subsample, 20000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata = run_STitch3D(ann_list)
        
    elif env_name == 'spacel':
        print('### Using SPACEL environment')
        from run_SPACEL import run_SPACEL
        ann_list = subsample_adata(ann_list, subsample, 20000)
        adata = run_SPACEL(ann_list, sample_list, output_path)

    elif env_name == 'scslat':
        print('### Using SLAT environment')
        from run_scSLAT import run_scSLAT
        ann_list = subsample_adata(ann_list, subsample, 20000)
        adata, pi_list, matching_cell_ids_list = run_scSLAT(ann_list,peak_data)

    elif env_name == 'sedr':
        print('### Using SEDR environment')
        from run_SEDR import run_SEDR
        ann_list = subsample_adata(ann_list, subsample, 20000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata = run_SEDR(ann_list, sample_list, cluster_option)

    elif env_name == 'spiral':
        print('### Using SPIRAL environment')
        from run_SPIRAL import run_SPIRAL
        ann_list = subsample_adata(ann_list, subsample, 5000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        ann_list = filter_common_genes(ann_list)
        adata = run_SPIRAL(ann_list, sample_list, output_path)
    
    elif env_name == 'stamp':
        print('### Using STAMP environment')
        from run_STAMP import run_STAMP
        ann_list = subsample_adata(ann_list, subsample, 20000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata = run_STAMP(ann_list, cluster_option)
        
    elif env_name == 'moscot':
        print('### Using MOSCOT environment')
        from run_moscot_NR import run_moscot_NR
        ann_list = subsample_adata(ann_list, subsample, 20000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata, pi_list, matching_cell_ids_list = run_moscot_NR(ann_list)
        
    elif env_name == 'moscot':
        print('### Using MOSCOT environment')
        from run_moscot_R import run_moscot_R
        ann_list = subsample_adata(ann_list, subsample, 20000)
        for ann in ann_list:
            sc.pp.filter_cells(ann, min_genes=1)
            sc.pp.filter_genes(ann, min_cells=1)
        adata, pi_list, matching_cell_ids_list = run_moscot_R(ann_list)

    else:
        raise ValueError(f"Unknown or unsupported environment: {env_name}")

    #################### Evaluation ####################   
    print('### Evaluating...')
    run_time = float(time.time() - start)
    print(f"### Run time: {run_time} seconds")
    stop_event.set()
    monitor_thread.join()  
    memory_usage = max_cpu_memory + max_gpu_memory
    print(f"### Memory usage: {memory_usage} MB")
    print(f"### Max CPU memory: {max_cpu_memory} MB")
    print(f"### Max GPU memory: {max_gpu_memory} MB")

    metrics_df = evaluation(adata, 
                            metrics_list=metrics_list,      
                            cluster_option=cluster_option, 
                            used_time=run_time, 
                            used_cpu_memory=max_cpu_memory,
                            used_gpu_memory=max_gpu_memory,
                            used_memory=memory_usage,
                            spatial_key='spatial',
                            spatial_aligned_key = 'spatial_aligned',
                            angle_true=angle_true,
                            matching_list=matching_cell_ids_list,
                            matching_ground_truth_list=matching_ground_truth_list,
                            pi_list = pi_list,
                            peak_data = peak_data
                            )
    print('### Saving...')
    tmp = sc.AnnData(obs=adata.obs.copy(), obsm=adata.obsm.copy())
    
    # # Save metrics
    metrics_df.to_csv(os.path.join(output_path, 'metrics.csv'), index=False)

    # Save meta info
    meta_info = pd.DataFrame(adata.obs)
    meta_info.to_csv(os.path.join(output_path, 'meta_info.csv'), index=True)

    # Save spatial raw
    spatial_raw = pd.DataFrame(adata.obsm['spatial'])
    spatial_raw.columns = ['x', 'y']
    spatial_raw.batch = adata.obs['batch']
    spatial_raw.to_csv(os.path.join(output_path, 'spatial.csv'), index=True)
        
    if 'mapping' in metrics_list:
        # Save spatial aligned
        spatial_aligned = pd.DataFrame(adata.obsm['spatial_aligned'])
        spatial_aligned.columns = ['x', 'y']
        spatial_aligned.batch = adata.obs['batch']
        spatial_aligned.to_csv(os.path.join(output_path, 'spatial_aligned.csv'), index=True)
        
    if 'embedding' in metrics_list:
        # Save embedding
        emb = pd.DataFrame(adata.obsm['integrated'])
        emb.batch = adata.obs['batch']
        emb.to_csv(os.path.join(output_path, 'embedding.csv'), index=True)
        
        if tmp.obsm['integrated'].shape[1] < 30:
            sc.pp.neighbors(tmp, n_neighbors=10, n_pcs=tmp.obsm['integrated'].shape[1],use_rep='integrated')
        else:
            sc.pp.neighbors(tmp, n_neighbors=10, n_pcs=30,use_rep='integrated')
        sc.tl.umap(tmp)
        
    if 'matching' in metrics_list:
        for i in range(len(matching_cell_ids_list)):
            matching_cell_ids_list[i].to_csv(os.path.join(output_path, f'matching_{sample_list[i]}_{sample_list[i+1]}.csv'), index=False)
        # Save pi
        for i in range(len(pi_list)):
            pi_list[i].to_csv(os.path.join(output_path, f'pi_{sample_list[i]}_{sample_list[i+1]}.csv'), index=False)
    
    tmp.write(os.path.join(output_path, 'adata.h5ad'))

    def cleanup_memory(*args):
        import gc
        for obj in args:
            del obj
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except ImportError:
            pass
    cleanup_memory(adata, ann_list, metrics_df, pi_list, tmp)
        

if rep == 1:
    run_benchmarking(input_path,
                    base_output_path,
                    sample_list,
                    cluster_option,
                    metrics_list,
                    rigid_option,
                    angle_true = angle_true,
                    overlap_list = overlap_list,
                    distortion_list = distortion_list
                    # pseudocount = pseudocount,
                    # rep_num = None
                    )
else:
    for rep_num in range(1,rep+1):
        run_benchmarking(input_path,
                        base_output_path,
                        sample_list,
                        cluster_option,
                        metrics_list,
                        rigid_option,
                        angle_true = angle_true,
                        overlap_list = overlap_list,
                        distortion_list = distortion_list
                        # pseudocount = pseudocount,
                        # rep_num = rep_num
                        )
