import os
import numpy as np
import h5py
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from tqdm import tqdm


def get_uniprot_ids(af_folder="data/AlphaFoldData/"):
    uniprot_ids = {x.split("-")[1] for x in os.listdir(af_folder) if "AF-" in x}
    sorted_ids = sorted(uniprot_ids)
    print("{} unique uniprot_ids.".format(len(sorted_ids)))
    return sorted_ids

def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as h5file:
        return np.array(h5file['atom_coordinates'])

def save_to_hdf5(csr_graph, features, file_path):
    with h5py.File(file_path, 'w') as f:
        # Save the features matrix
        f.create_dataset('features', data=features)
        # Save the CSR components
        f.create_dataset('data', data=csr_graph.data)
        f.create_dataset('indices', data=csr_graph.indices)
        f.create_dataset('indptr', data=csr_graph.indptr)
        f.create_dataset('shape', data=csr_graph.shape)


def csr_graph_from_point_cloud(atom_point_cloud, STANDARD_BOND_LENGTH=1.5):
    # Build a k-d tree for quick nearest-neighbor lookup
    tree = cKDTree(atom_point_cloud[:,1:])
    
    # Query the tree for pairs within the bond length
    pairs = tree.query_pairs(r=STANDARD_BOND_LENGTH)
    
    # Create row index and column index arrays for CSR format
    row_idx = np.array([pair[0] for pair in pairs])
    col_idx = np.array([pair[1] for pair in pairs])
    
    # Create data array for CSR format (all ones, assuming single bond)
    data = np.ones(len(pairs))
    
    # Create the CSR matrix
    csr_graph = csr_matrix((data, (row_idx, col_idx)), shape=(len(atom_point_cloud), len(atom_point_cloud)))
    
    return csr_graph


def get_atom_cloud_filenames(data_path, uniprot_ids):
    return ["AF-{}-F1-model_v4_atom_cloud.hdf5".format(x) for x in uniprot_ids]
    

def main():
    data_path = "data/protein_atom_point_clouds"
    output_dir_path = "data/protein_atom_graphs"

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)


    unique_atom_cloud_files = get_atom_cloud_filenames(data_path, get_uniprot_ids())

    for file_name in tqdm(unique_atom_cloud_files, desc='Processing files', unit='file'):
        file_path = os.path.join(data_path, file_name)

        atom_point_cloud = read_hdf5(file_path)

        # Assuming the atom types are in the first column of the point cloud array
        atom_point_cloud_atom_types = atom_point_cloud[:, 0]  # Changed from :1 to 0 for correct indexing
        n_atom_types = 9

        # One-hot encode the atom types
        features = np.eye(n_atom_types)[atom_point_cloud_atom_types.astype(int) - 1]  # Make sure the types are integers

        # Now features is a two-dimensional numpy array with one-hot encoding

        csr_graph = csr_graph_from_point_cloud(atom_point_cloud)

        new_file_name = file_name.replace('_atom_cloud', '_graph')

        output_file_path = os.path.join(output_dir_path, new_file_name)
        save_to_hdf5(csr_graph, features, output_file_path)
        


if __name__ == "__main__":
    main()