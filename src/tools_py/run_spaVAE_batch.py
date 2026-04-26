import warnings
warnings.filterwarnings("ignore")

from time import time
import torch
from spaVAE_Batch import kernel, SVGP_Batch
from spaVAE_Batch import SPAVAE
from sklearn.preprocessing import MinMaxScaler
import scanpy as sc
from spaVAE_Batch.preprocess import normalize
import numpy as np
import pandas as pd
import argparse
import gc
from utils import get_ann_list, create_lightweight_adata,set_seed
from clustering import clustering


# torch.manual_seed(42)

def run_spaVAE_batch(adata, **args_dict):
    seed = args_dict['seed']
    set_seed(seed)
    clust = args_dict['clust']
    
    ann_list, config = get_ann_list(adata)
    del adata
    gc.collect()
    
    lr = 1e-3
    if 'data_id' in ann_list[0].obs.columns:
        data_id = ann_list[0].obs['data_id'].unique()[0]
        if data_id == 'SD6':
            lr =  1e-4
            print(f"Detected SD6: Scaling down learning rate to {lr}")

    parser = argparse.ArgumentParser(description='Spatial dependency-aware variational autoencoder for integrating batches of data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--batch_size', default="auto")
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--train_size', default=0.95, type=float)
    parser.add_argument('--patience', default=200, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--noise', default=0, type=float)
    parser.add_argument('--dropoutE', default=0, type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0, type=float,
                        help='dropout probability for decoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[128, 64], type=int)
    parser.add_argument('--GP_dim', default=2, type=int,help='dimension of the latent Gaussian process embedding')
    parser.add_argument('--Normal_dim', default=8, type=int,help='dimension of the latent standard Gaussian embedding')
    parser.add_argument('--decoder_layers', nargs="+", default=[128], type=int)
    parser.add_argument('--dynamicVAE', default=True, type=bool, 
                        help='whether to use dynamicVAE to tune the value of beta, if setting to false, then beta is fixed to initial value')
    parser.add_argument('--init_beta', default=10, type=float, help='initial coefficient of the KL loss')
    parser.add_argument('--min_beta', default=4, type=float, help='minimal coefficient of the KL loss')
    parser.add_argument('--max_beta', default=25, type=float, help='maximal coefficient of the KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--shared_dispersion', default=False, type=bool)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--inducing_point_steps', default=6, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--allow_batch_kernel_scale', default=True, type=bool)
    parser.add_argument('--model_file', default='model.pt')
    parser.add_argument('--final_latent_file', default='final_latent.txt')
    parser.add_argument('--denoised_counts_file', default='denoised_counts.txt')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    def filter_shared_hvg_list(ann_list, n_top_genes=3000):
        hvg_sets = []
        
        for idx, adata in enumerate(ann_list):
            n_genes = adata.shape[1]
            
            if n_genes <= n_top_genes:
                hvg = set(adata.var_names)
            else:
                try:
                    sc.pp.highly_variable_genes(
                        adata, 
                        n_top_genes=n_top_genes, 
                        flavor='seurat_v3', 
                        inplace=True
                    )
                    hvg = set(adata.var_names[adata.var['highly_variable']])
                except Exception:
                    hvg = set(adata.var_names)
            
            hvg_sets.append(hvg)
            gc.collect()

        if not hvg_sets:
            return []

        shared_genes = sorted(set.intersection(*hvg_sets))
        print(f"Number of shared HVGs across all samples: {len(shared_genes)}")

        filtered_list = [adata[:, shared_genes].copy() for adata in ann_list]

        return filtered_list

    for ann in ann_list:
        ann.obs['cell_id'] = ann.obs_names
        ann.layers['counts'] = ann.X
        
    ann_list = filter_shared_hvg_list(ann_list)
    x_list = []
    loc_list = []
    batch_list = []
    meta_list = []

    for i, adata in enumerate(ann_list):

        x = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        x_list.append(x)

        loc = adata.obsm["spatial"]
        loc_list.append(loc)

        b = np.zeros((adata.shape[0], len(ann_list)))
        b[:, i] = 1
        batch_list.append(b)

        meta_list.append(adata.obs[['Ground Truth','cell_id']])
        
    del ann_list
    gc.collect()

    x = np.concatenate(x_list, axis=0).astype('float32')       # shape: [total_spots, genes]
    loc = np.concatenate(loc_list, axis=0).astype('float32')      # shape: [total_spots, 2]
    batch = np.concatenate(batch_list, axis=0).astype('float32')   # shape: [total_spots, n_batch]

    original_loc = np.concatenate(loc_list, axis=0)

    meta = pd.concat(meta_list, axis=0)
    
    if args.batch_size == "auto":
        if x.shape[0] <= 1024:
            args.batch_size = 128
        elif x.shape[0] <= 2048:
            args.batch_size = 256
        else:
            args.batch_size = 512
    else:
        args.batch_size = int(args.batch_size)

    n_batch = batch.shape[1]

    # scale locations per batch
    loc_scaled = np.zeros(loc.shape, dtype=np.float32)
    for i in range(n_batch):
        scaler = MinMaxScaler()
        b_loc = loc[batch[:,i]==1, :]
        b_loc = scaler.fit_transform(b_loc) * args.loc_range
        loc_scaled[batch[:,i]==1, :] = b_loc
    loc = loc_scaled

    loc = np.concatenate((loc, batch), axis=1)

    # build inducing point matrix with batch index
    eps = 1e-5
    initial_inducing_points_0_ = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
    initial_inducing_points_0 = np.tile(initial_inducing_points_0_, (n_batch, 1))
    initial_inducing_points_1 = []
    for i in range(n_batch):
        initial_inducing_points_1_ = np.zeros((initial_inducing_points_0_.shape[0], n_batch))
        initial_inducing_points_1_[:, i] = 1
        initial_inducing_points_1.append(initial_inducing_points_1_)
    initial_inducing_points_1 = np.concatenate(initial_inducing_points_1, axis=0)
    initial_inducing_points = np.concatenate((initial_inducing_points_0, initial_inducing_points_1), axis=1)
    print(initial_inducing_points.shape)

    meta.index = meta['cell_id'].values
    adata = sc.AnnData(x, dtype="float32", obs=meta)
    adata.obsm['input_loc'] = loc      
    adata.obsm['input_batch'] = batch  
    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    
    loc = adata.obsm['input_loc']
    batch = adata.obsm['input_batch']

    model = SPAVAE(input_dim=adata.n_vars, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, n_batch=n_batch, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD, shared_dispersion=args.shared_dispersion,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, allow_batch_kernel_scale=args.allow_batch_kernel_scale,
        N_train=adata.n_obs, KL_loss=args.KL_loss, dynamicVAE=args.dynamicVAE, init_beta=args.init_beta, min_beta=args.min_beta, max_beta=args.max_beta, 
        dtype=torch.float32, device=args.device)
    model = model.to(args.device)

    t0 = time()
    model.train_model(pos=loc, ncounts=adata.X, raw_counts=adata.raw.X, size_factors=adata.obs.size_factors, batch=batch,
                lr=lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=True, model_weights=args.model_file)
    print('Training time: %d seconds.' % int(time() - t0))

    final_latent = model.batching_latent_samples(X=loc, Y=adata.X, B=batch, batch_size=args.batch_size)
    adata.obsm['integrated'] = final_latent
    
    # Clustering
    print('### Clustering...')
    adata = clustering(adata, use_rep='integrated', label_key='Ground Truth', method=clust)
    adata.obs['benchmark_cluster'] = adata.obs[clust]
    adata.obs['batch'] = np.argmax(batch, axis=1).astype(str)
    adata.obsm['spatial'] = original_loc
    
    adata = create_lightweight_adata(adata, config, args_dict=args_dict)
    
    return adata
    
    
    
    
    