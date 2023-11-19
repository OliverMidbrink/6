import h5py
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib.pyplot as plt

def load_hdf5_to_numpy(file_path="interactome/HuRI_to_Alphafold_PPI_adj_matrix.h5", dataset_name="HuRI_to_Alphafold_PPI_adj_matrix"):
    """
    Load data from an HDF5 file into a NumPy array.

    Parameters:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset in the HDF5 file.

    Returns:
        numpy.ndarray: Loaded data as a NumPy array.
    """
    with h5py.File(file_path, 'r') as file:
        dataset = file[dataset_name]
        return np.array(dataset)



adj_matrix = load_hdf5_to_numpy()
csr_matrix = csr_matrix(adj_matrix)

with h5py.File('interactome/HuRI_to_Alphafold_PPI_csr_matrix.h5', 'w') as file:
    # Create a dataset and save the CSR matrix
    print(csr_matrix.data)
    print(csr_matrix.shape)
    dataset = file.create_dataset('csr_matrix', data=csr_matrix.data, shape=csr_matrix.shape, dtype='f')
    dataset.attrs['indices'] = csr_matrix.indices
    dataset.attrs['indptr'] = csr_matrix.indptr
