from itertools import product, islice, chain
import math
import multiprocessing as mp
import os.path
import time

import h5py
import numpy as np
from tqdm import tqdm

 
def _chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def compute_pairswise_distance(X, num_samples:int, temp_filename:str, num_chunks:int = 100, del_tempfile:bool = False):
    # num_samples: number of samples in X
    # temp_filename: filename for temp HDF5 file created
    # num_chunks: number of chunks to split dataset into for processing per worker 
    # del_tempfile: whether to delete temp HDF5 file afterwards
    
    # _start = time.time()
    filename = f'./{temp_filename}.hdf'
    
    if os.path.exists(filename):
        print("HDF5 file {filename} for pairwise distance already exists, skipping this step!")
    else:
        print(f"Generating pairwise distance, writing to HDF5 file {filename} ...")
        _generator = product(X, repeat=2)
        chunk_size = math.ceil(num_samples**2 / num_chunks)

        h5f = h5py.File(filename, 'a')
        for _chunk in tqdm(_chunks(_generator, chunk_size), total=num_chunks):
            distances = np.square([np.subtract(x[0], x[1]) for x in _chunk]).sum(-1)
            if "dataset" not in h5f:
                h5f_dataset = h5f.create_dataset('dataset', data=distances, compression="gzip", chunks=True, maxshape=(None, )) 
            else:
                h5f_dataset.resize((h5f_dataset.shape[0] + distances.shape[0]), axis = 0)
                h5f_dataset[-distances.shape[0]:] = distances
        h5f.close()

    distances = []
    with h5py.File(filename,'r') as infile:
        distances = infile['dataset'][:]
    distances = distances.reshape(num_samples, num_samples)
    
    if del_tempfile and os.path.exists(filename): os.remove(filename)

    # _elapsedTime = (time.time() - _start) / 60
    # print("Total time taken: ", _elapsedTime)
    return distances

    # 10000 samples **2 will take 20mins /w 100 chunks
        # hits around 9gb RAM usage, double 'num_chunks' if MemoryError
        # 350MB HDF5 file, 400MB RAM
        # 3 hr training time?
    # 60000 samples **2 will take 36 times longer i.e. 12hr
        # probably 12.6gb HDF5 file, 14,4gb RAM
