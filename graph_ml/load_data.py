import os, sys
import numpy as np
import h5py
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from tqdm import tqdm

def load_graph_from_hdf5(file_path):
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

def load_one_hot_vector(one_hot_dir, file_name):
    # Constructs the full path to the HDF5 file
    file_path = os.path.join(one_hot_dir, file_name)
    # Uses the h5py library to load the one-hot vector from the HDF5 file
    with h5py.File(file_path, 'r') as h5f:
        one_hot_vector = h5f['one_hot'][:]
    return one_hot_vector


def load_data(graph_data_dir_path="data/protein_atom_graphs", alphabetic_id_one_hot_data_dir_path="data/protein_one_hot_id_vectors"):
    x_data, y_data = [], []

    for file_name in tqdm(sorted(os.listdir(graph_data_dir_path))[:200], desc='Loading training data', unit='file'):
        graph_file_path = os.path.join(graph_data_dir_path, file_name)
        one_hot_filename = file_name.replace("_graph", "_alphabetic_one_hot_id")

        csr_graph, feature = load_graph_from_hdf5(graph_file_path)
        one_hot_id_vector = load_one_hot_vector(alphabetic_id_one_hot_data_dir_path, one_hot_filename)

        x_data.append((csr_graph, feature))
        y_data.append(one_hot_id_vector)

    return x_data, y_data

def main():
    graph_data_dir_path = "data/protein_atom_graphs"
    alphabetic_id_one_hot_data_dir_path = "data/protein_one_hot_id_vectors"

    x_train, y_train = load_data(graph_data_dir_path, alphabetic_id_one_hot_data_dir_path)

    # Example print statements to verify data loading
    print(f"Total training examples: {len(x_train)}")
    print(f"Total labels: {len(y_train)}")

if __name__ == "__main__":
    main()