import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter

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

pooling_window_size = (50, 50)

# Perform max-pooling
max_pooled_array = maximum_filter(adj_matrix, size=pooling_window_size)




plt.imshow(max_pooled_array, cmap='gray')  # Use 'gray' colormap for grayscale images
plt.axis('off')  # Turn off axis labels and ticks
plt.show()
