# load_data.py
import os
import numpy as np
import h5py
from scipy.sparse import csr_matrix
from spektral.data import Graph, Dataset
from tqdm import tqdm

class ProteinGraphDataset(Dataset):
    def __init__(self, graph_data_dir_path, alphabetic_id_one_hot_data_dir_path, **kwargs):
        self.graph_data_dir_path = graph_data_dir_path
        self.alphabetic_id_one_hot_data_dir_path = alphabetic_id_one_hot_data_dir_path
        super().__init__(**kwargs)

    def read(self):
        graphs = []
        for file_name in tqdm(sorted(os.listdir(self.graph_data_dir_path))[:20], desc='Loading graphs', unit='graph'):
            graph_file_path = os.path.join(self.graph_data_dir_path, file_name)
            one_hot_filename = file_name.replace("_graph", "_alphabetic_one_hot_id")

            csr_graph, feature = self.load_graph_from_hdf5(graph_file_path)
            one_hot_id_vector = self.load_one_hot_vector(one_hot_filename)

            self.n_classes = len(one_hot_id_vector)

            # Create Graph object
            graph = Graph(x=feature, a=csr_graph, y=one_hot_id_vector)
            graphs.append(graph)

        return graphs

    def load_graph_from_hdf5(self, file_path):
        with h5py.File(file_path, 'r') as f:
            # Load the features matrix
            features = f['features'][:]
            # Load the CSR components
            data = f['data'][:]
            indices = f['indices'][:]
            indptr = f['indptr'][:]
            shape = f['shape'][:]
            # Reconstruct the CSR matrix
            csr_graph = csr_matrix((data, indices, indptr), shape=shape)
        return csr_graph, features

    def load_one_hot_vector(self, file_name):
        # Constructs the full path to the HDF5 file
        file_path = os.path.join(self.alphabetic_id_one_hot_data_dir_path, file_name)
        # Uses the h5py library to load the one-hot vector from the HDF5 file
        with h5py.File(file_path, 'r') as h5f:
            one_hot_vector = h5f['one_hot'][:]
        return one_hot_vector