import h5py
from scipy.sparse import csr_matrix
from spektral.data import Graph
from tensorflow.keras.utils import Sequence
import numpy as np
import os

def load_graph_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        # Load the features matrix
        features = f['features'][:]

        # Load the CSR components and reconstruct the adjacency matrix
        data = f['data'][:]
        indices = f['indices'][:]
        indptr = f['indptr'][:]
        shape = f['shape'][:]
        adjacency = csr_matrix((data, indices, indptr), shape=shape)

        # Load the labels or targets if they exist
        label = f['labels'] if 'labels' in f else None

    # Create the Spektral Graph object
    graph = Graph(x=features, a=adjacency, y=label)

    return graph



def train_generator():
    train_data_path = "data/iPPI_graphs/train_graphs"
    train_files = sorted(os.listdir(train_data_path))

    for train_file in train_files:
        graph = load_graph_from_hdf5(os.path.join(train_data_path, train_file))

        yield [graph]