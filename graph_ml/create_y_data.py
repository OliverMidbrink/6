import os
import numpy as np
import h5py
from tqdm import tqdm

def save_one_hot_vector(one_hot_vector, output_dir, file_name):
    # Use the h5py library to save the one-hot vector to an HDF5 file
    with h5py.File(os.path.join(output_dir, file_name), 'w') as h5f:
        h5f.create_dataset('one_hot', data=one_hot_vector)

def main():
    graph_data_dir_path = "data/protein_atom_graphs"
    one_hot_dir_path = "data/protein_one_hot_id_vectors"

    # Ensure the output directory exists
    if not os.path.exists(one_hot_dir_path):
        os.makedirs(one_hot_dir_path)

    # List files in the directory and sort them alphabetically
    files = sorted(os.listdir(graph_data_dir_path))

    # Assign a unique integer index to each filename
    file_to_index = {file: idx for idx, file in enumerate(files)}

    # Iterate over files and create one-hot vectors
    for file_name in tqdm(files, desc='Creating One Hot Protein ID files', unit='file'):
        # Create a one-hot encoded vector for the current file
        one_hot_vector = np.zeros(len(files), dtype=int)
        one_hot_vector[file_to_index[file_name]] = 1
        
        # Save the one-hot vector to a new HDF5 file
        new_file_name = file_name.replace("_graph", "_alphabetic_one_hot_id")

        save_one_hot_vector(one_hot_vector, one_hot_dir_path, new_file_name)

if __name__ == "__main__":
    main()
