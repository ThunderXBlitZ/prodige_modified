# (Helper) functions for calculating pairwise distance for various compressions e.g. MDS, PCA

import gc
from itertools import combinations, islice, chain
import math
import multiprocessing as mp
import os.path
import time

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from tqdm import tqdm

 
def _chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def compute_original_pairwise_distance(X, simple:bool, num_samples:int = None, temp_filename:str = None, num_chunks:int = 100, del_tempfile:bool = True):
    if simple:
        distances = np.square(X[None, :, :] - X[:, None, :]).sum(-1) ** 0.5
    else:
        distances = pairwise_distance(X, num_samples, temp_filename, num_chunks, del_tempfile)
    return distances


def compute_pca_pairwise_distance(X, simple:bool, num_samples:int = None, n_comp:int=4, temp_filename:str = None, num_chunks:int = 100, del_tempfile:bool = True):
    pca = PCA(n_components=n_comp).fit(X)
    X_pca = pca.inverse_transform(pca.transform(X))
    if simple:
        pca_distances = np.square(X_pca[None, :, :] - X_pca[:, None, :]).sum(-1) ** 0.5
    else:
        pca_distances = pairwise_distance(X_pca, num_samples, temp_filename, num_chunks, del_tempfile)
    return pca_distances


def pairwise_distance(X, num_samples:int, temp_filename:str, num_chunks:int = 100, del_tempfile:bool = True):
    # num_samples: number of samples
    # temp_filename: filename for temp HDF5 file created
    # num_chunks: number of chunks to split dataset into for processing per worker 
    # del_tempfile: whether to delete temp HDF5 file afterwards
    
    filename = f'./data/{temp_filename}.hdf'
    
    if os.path.exists(filename):
        print("HDF5 file {filename} for pairwise distance already exists, skipping this step!")
    else:
        _iterator = combinations(X, 2)
        chunk_size = math.ceil((num_samples**2 / 2 - num_samples) / num_chunks)
        print(f"Generating pairwise distance, writing to HDF5 file {filename} ...")
        h5f = h5py.File(filename, 'a')
        for _chunk in tqdm(_chunks(_iterator, chunk_size), total=num_chunks):
            _temp = np.array([[x[0], x[1]] for x in _chunk])
            _temp = _temp.swapaxes(0,1)
            distances = np.square(np.subtract(_temp[0], _temp[1])).sum(-1) ** 0.5
            if "dataset" not in h5f:
                h5f_dataset = h5f.create_dataset('dataset', data=distances, compression="gzip", chunks=True, maxshape=(None, )) 
            else:
                h5f_dataset.resize((h5f_dataset.shape[0] + distances.shape[0]), axis = 0)
                h5f_dataset[-distances.shape[0]:] = distances
            gc.collect()
        h5f.close()
        print("Successfully written results to HDF5 file.")

    distances = []
    with h5py.File(filename,'r') as infile:
        distances = infile['dataset'][:]

    output = np.zeros(shape=(num_samples, num_samples))
    u_ids = np.triu_indices(num_samples, 1)
    output[u_ids] = distances
    distances = output + output.T
    
    if del_tempfile and os.path.exists(filename):
        os.remove(filename)

    return distances


def mds_pairwise_distance(X, n_comp:int = 4, n_jobs: int = -1):
    mds = MDS(n_components=4, random_state=42, n_jobs=-1, dissimilarity='euclidean')
    X_mds = mds.fit_transform(X)
    mds_distances = np.square(X_mds[None, :, :] - X_mds[:, None, :]).sum(-1) ** 0.5
    return mds_distances