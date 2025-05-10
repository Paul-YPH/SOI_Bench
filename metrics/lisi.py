###############################################
################## Embedding ##################
###############################################

# The code is modified from https://github.com/theislab/scib/blob/main/scib/metrics/lisi.py
# /net/mulan/home/penghuy/benchmark/metrics/knn_graph
# Need to make /net/mulan/home/penghuy/benchmark/metrics/knn_graph/knn_graph.o with :g++ -std=c++11 -O3 knn_graph.cpp -o knn_graph.o


import itertools
import multiprocessing as mp
import os
import pathlib
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from numba import njit
from scipy.io import mmwrite



def compute_scaled_lisi(
    adata,
    key,
    scale=True,
    compute_fn=None,
    type="embed",
    use_rep="integrated",
    k0=90,
    subsample=None,
    n_cores=1,
    verbose=False,
):
    """
    Compute and optionally scale the LISI score.
    """
    score = compute_lisi(adata, key, type, use_rep, k0, subsample, n_cores, verbose)
    if scale:
        n_unique = adata.obs[key].nunique()
        score = (n_unique - score) / (n_unique - 1) if compute_fn == 'clisi' else (score - 1) / (n_unique - 1)
    return score


def compute_ilisi(
    adata,
    batch_key,
    scale=True,
    type="embed",
    use_rep="integrated",
    k0=90,
    subsample=None,
    n_cores=1,
    verbose=False,
    **kwargs
):
    """
    Compute the iLISI score for batch mixing.
    """
    results = []
    value = compute_scaled_lisi(
        adata,
        batch_key,
        scale=scale,
        compute_fn='ilisi',
        type=type,
        use_rep=use_rep,
        k0=k0,
        subsample=subsample,
        n_cores=n_cores,
        verbose=verbose,
        **kwargs,
    )
    results.append({'metric': 'ilisi', 'value': value, 'group': batch_key})
    df = pd.DataFrame(results)
    return df


def compute_clisi(
    adata,
    label_key,
    scale=True,
    type="embed",
    use_rep="integrated",
    k0=90,
    subsample=None,
    n_cores=1,
    verbose=False,
    **kwargs
):
    """
    Compute the cLISI score for batch mixing.
    """
    results = []
    value = compute_scaled_lisi(
        adata,
        label_key,
        scale=scale,
        compute_fn='clisi',
        type=type,
        use_rep=use_rep,
        k0=k0,
        subsample=subsample,
        n_cores=n_cores,
        verbose=verbose,
        **kwargs,
    )
    results.append({'metric': 'clisi', 'value': value, 'group': label_key})
    df = pd.DataFrame(results)
    return df

def compute_lisi_f1(ilisi_df, clisi_df):
    """
    Compute a balanced F1-like score combining iLISI and cLISI.

    :param ilisi: iLISI score (higher is better for batch mixing).
    :param clisi: cLISI score (higher is better for cell type separation).
    :return: Computed F1-like score.
    """
    ilisi = ilisi_df['value'].values
    clisi = clisi_df['value'].values
    # Ensure inputs are within valid range
    if not (0 <= ilisi <= 1 and 0 <= clisi <= 1):
        raise ValueError("iLISI and cLISI must be in the range [0, 1].")
    
    # Compute the score
    part1 = 2 * clisi * ilisi
    part2 = clisi + ilisi
    
    # Avoid division by zero
    if part2 == 0:
        return 0  # Or handle this case differently if required
    
    f1_score = part1 / part2
    df = pd.DataFrame({'metric': 'lisi_f1', 'value': f1_score, 'group': 'f1'})
    return df

def compute_lisi(
    adata,
    batch_key,
    type="embed",
    use_rep="integrated",
    k0=90,
    subsample=None,
    n_cores=1,
    verbose=False,
):
    adata = recompute_knn(adata, type, use_rep)
    lisi_score = lisi_graph_py(
        adata=adata,
        obs_key=batch_key,
        n_neighbors=k0,
        perplexity=None,
        subsample=subsample,
        n_cores=n_cores,
        verbose=verbose,
    )

    lisi = np.nanmedian(lisi_score)

    return lisi


def recompute_knn(adata, type, use_rep):
    """Recompute neighbours"""
    if type == "embed":
        return sc.pp.neighbors(adata, n_neighbors=15, use_rep=use_rep, copy=True)
    elif type == "full":
        if "X_pca" not in adata.obsm.keys():
            sc.pp.pca(adata, svd_solver="arpack")
        return sc.pp.neighbors(adata, n_neighbors=15, copy=True)
    else:
        # if knn - do not compute a new neighbourhood graph (it exists already)
        return adata.copy()

def lisi_graph_py(
    adata,
    obs_key,
    n_neighbors=90,
    perplexity=None,
    subsample=None,
    n_cores=1,
    verbose=False,
):
    # use no more than the available cores
    n_cores = max(1, min(n_cores, mp.cpu_count()))

    if "neighbors" not in adata.uns:
        raise AttributeError(
            "Key 'neighbors' not found. Please make sure that a kNN graph has been computed"
        )
    elif verbose:
        print("using precomputed kNN graph")

    # get knn index matrix
    if verbose:
        print("Convert nearest neighbor matrix and distances for LISI.")

    adata.obs[obs_key] = adata.obs[obs_key].astype("category")
    batch_labels = adata.obs[obs_key].cat.codes.values
    n_batches = adata.obs[obs_key].nunique()

    if perplexity is None or perplexity >= n_neighbors:
        # use LISI default
        perplexity = np.floor(n_neighbors / 3)

    # setup subsampling
    subset = 100  # default, no subsampling
    if subsample is not None:
        subset = subsample  # do not use subsampling
        if isinstance(subsample, int) is False:  # need to set as integer
            subset = int(subsample)

    # run LISI in python
    if verbose:
        print("Compute knn on shortest paths")

    # set connectivities to 3e-308 if they are lower than 3e-308 (because cpp can't handle double values smaller than that).
    connectivities = adata.obsp["connectivities"]  # csr matrix format
    large_enough = connectivities.data >= 3e-308
    if verbose:
        n_too_small = np.sum(large_enough is False)
        if n_too_small:
            print(
                f"{n_too_small} connectivities are smaller than 3e-308 and will be set to 3e-308"
            )
            print(connectivities.data[large_enough is False])
    connectivities.data[large_enough is False] = 3e-308

    # temporary file
    with tempfile.TemporaryDirectory(prefix="lisi_") as tmpdir:
        prefix = f"{tmpdir}/graph_lisi"
        mtx_file_path = prefix + "_input.mtx"
        mmwrite(mtx_file_path, connectivities, symmetry="general")

        # call knn-graph computation in Cpp
        root = pathlib.Path(__file__).parent  # get current root directory
        cpp_file_path = (
            root / "knn_graph/knn_graph.o"
        )  # create POSIX path to file to execute compiled cpp-code
        # comment: POSIX path needs to be converted to string - done below with 'as_posix()'
        # create evenly split chunks if n_obs is divisible by n_chunks (doesn't really make sense on 2nd thought)
        args_int = [
            cpp_file_path.as_posix(),
            mtx_file_path,
            prefix,
            str(n_neighbors),
            str(n_cores),  # number of splits
            str(subset),
        ]

        try:
            if verbose:
                print(f'call {" ".join(args_int)}')
            subprocess.run(args_int)
        except RuntimeError as ex:
            print(f"Error computing LISI kNN graph {ex}\nSetting value to np.nan")
            return np.nan

        if verbose:
            print("LISI score estimation")

        if n_cores > 1:
            if verbose:
                print(f"{n_cores} processes started.")
            pool = mp.Pool(processes=n_cores)
            chunk_no = np.arange(0, n_cores)

            # create argument list for each worker
            results = pool.starmap(
                compute_simpson_index_graph,
                zip(
                    itertools.repeat(prefix),
                    itertools.repeat(batch_labels),
                    itertools.repeat(n_batches),
                    itertools.repeat(n_neighbors),
                    itertools.repeat(perplexity),
                    chunk_no,
                ),
            )
            pool.close()
            pool.join()

            simpson_estimate_batch = np.concatenate(results)

        else:
            simpson_estimate_batch = compute_simpson_index_graph(
                file_prefix=prefix,
                batch_labels=batch_labels,
                n_batches=n_batches,
                perplexity=perplexity,
                n_neighbors=n_neighbors,
            )

    return 1 / simpson_estimate_batch


# LISI core functions


def compute_simpson_index_graph(
    file_prefix=None,
    batch_labels=None,
    n_batches=None,
    n_neighbors=90,
    perplexity=30,
    chunk_no=0,
    tol=1e-5,
):
    index_file = file_prefix + "_indices_" + str(chunk_no) + ".txt"
    distance_file = file_prefix + "_distances_" + str(chunk_no) + ".txt"

    # check if the target file is not empty
    if os.stat(index_file).st_size == 0:
        print("File has no entries. Doing nothing.")
        lists = np.zeros(0)
        return lists

    # read distances and indices with nan value handling
    header = ["index"] + list(range(1, n_neighbors + 1))
    indices = pd.read_table(index_file, index_col=0, header=None, sep=",", names=header)
    indices = indices.T

    distances = pd.read_table(
        distance_file, index_col=0, header=None, sep=",", names=header
    )
    distances = distances.T

    # get cell ids
    chunk_ids = indices.columns.values.astype("int")

    # initialize
    logU = np.log(perplexity)
    simpson = np.zeros(len(chunk_ids))

    # loop over all cells in chunk
    for i, chunk_id in enumerate(chunk_ids):
        # get neighbors and distances
        # read line i from indices matrix
        get_col = indices[chunk_id]

        if get_col.isnull().sum() > 0:
            # not enough neighbors
            print(f"Chunk {chunk_id} does not have enough neighbors. Skipping...")
            simpson[i] = 1  # np.nan #set nan for testing
            continue

        knn_idx = get_col.astype("int") - 1  # get 0-based indexing

        # read line i from distances matrix
        D_act = distances[chunk_id].values.astype("float")

        # start lisi estimation
        beta = 1
        betamin = -np.inf
        betamax = np.inf

        H, P = Hbeta(D_act, beta)
        Hdiff = H - logU
        tries = 0

        # first get neighbor probabilities
        while np.logical_and(np.abs(Hdiff) > tol, tries < 50):
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf:
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if betamin == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2

            H, P = Hbeta(D_act, beta)
            Hdiff = H - logU
            tries += 1

        if H == 0:
            simpson[i] = -1
            continue
            # then compute Simpson's Index
        batch = batch_labels[knn_idx]
        B = convert_to_one_hot(batch, n_batches)
        sumP = np.matmul(P, B)  # sum P per batch
        simpson[i] = np.dot(sumP, sumP)  # sum squares

    return simpson


@njit
def Hbeta(D_row, beta):
    """
    Helper function for simpson index computation
    """
    D_row[np.isnan(D_row)] = 0
    P = np.exp(-D_row * beta)
    sumP = np.sum(P)
    if sumP == 0:
        H = 0
        P = np.zeros(len(D_row))
    else:
        H = np.log(sumP) + beta * np.sum(D_row * P) / sumP
        P /= sumP
    return H, P


@njit
def convert_to_one_hot(vector, num_classes=None):
    if num_classes is None:
        num_classes = np.max(vector) + 1
    return np.eye(num_classes)[vector]