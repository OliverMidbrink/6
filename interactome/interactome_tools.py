from scipy.sparse import csr_matrix
import h5py
import os
import random

def load_csr_matrix(file_path):
    with h5py.File(file_path, 'r') as f:
        # Load the data, indices, and indptr from the file
        data = f['data'][:]
        indices = f['indices'][:]
        indptr = f['indptr'][:]
        shape = f.attrs['shape']
        # Construct and return the csr_matrix
        return csr_matrix((data, indices, indptr), shape=shape)

def get_uniprot_ids(af_folder="data/AlphaFoldData/"):
    uniprot_ids = {x.split("-")[1] for x in os.listdir(af_folder) if "AF-" in x}
    sorted_ids = sorted(uniprot_ids)
    print(len(sorted_ids))
    return sorted_ids


def get_neighbors(csr_matrix, uniprot_ids, uniprot_id, n_steps=1):
    # Create a mapping from UniProt IDs to indices
    uniprot_to_index = {uniprot_id: index for index, uniprot_id in enumerate(uniprot_ids)}
    # Create a reverse mapping from indices to UniProt IDs
    index_to_uniprot = {index: uniprot_id for uniprot_id, index in uniprot_to_index.items()}
    
    # Get the index for the given UniProt ID
    matrix_index = uniprot_to_index[uniprot_id]
    neighbors = set([matrix_index])
    
    # Perform breadth-first search to the n-th degree
    for _ in range(n_steps):
        current_neighbors = list(neighbors)
        # Use list comprehension to extract immediate neighbors for each node
        neighbors.update(
            index for node_index in current_neighbors
            for index in csr_matrix[node_index].indices
        )
    
    # Remove the original node to exclude it from its neighbors
    neighbors.remove(matrix_index)
    
    # Map indices back to UniProt IDs
    neighbor_uniprot_ids = [index_to_uniprot[n] for n in neighbors if n in index_to_uniprot]
    
    return neighbor_uniprot_ids

def get_interacting_uniprot_ids(csr_matrix, uniprot_id_list):
    random_uniprot = random.choice(uniprot_id_list)
    neighbors = get_neighbors(csr_matrix, uniprot_id_list, random_uniprot, n_steps=1)

    while len(neighbors) == 0:
        random_uniprot = random.choice(uniprot_id_list)
        neighbors = get_neighbors(csr_matrix, uniprot_id_list, random_uniprot, n_steps=1)
    
    return random_uniprot, random.choice(neighbors)

def get_non_interacting_uniprot_ids(csr_matrix, uniprot_id_list):
    random_uniprot = random.choice(uniprot_id_list)
    neighbors = get_neighbors(csr_matrix, uniprot_id_list, random_uniprot, n_steps=1)

    # Select a non-interacting uniprot ID by choosing one that's not a neighbor
    non_neighbor_uniprots = set(uniprot_id_list) - set(neighbors)
    while len(neighbors) != 0 and len(non_neighbor_uniprots) > 0:
        random_uniprot = random.choice(uniprot_id_list)
        neighbors = get_neighbors(csr_matrix, uniprot_id_list, random_uniprot, n_steps=1)
        non_neighbor_uniprots = set(uniprot_id_list) - set(neighbors)

    return random_uniprot, random.choice(list(non_neighbor_uniprots))


def main():
    csr_matrix = load_csr_matrix('interactome/HuRI_to_Alphafold_PPI_csr_matrix.h5')
    uniprot_ids_index = get_uniprot_ids()

    uniprots_connected = get_neighbors(csr_matrix, uniprot_ids_index, "O14718", n_steps=1)
    
    for x in range(1000):
        pair = get_non_interacting_uniprot_ids(csr_matrix, uniprot_ids_index)
        is_interacting = pair[0] in get_neighbors(csr_matrix, uniprot_ids_index, pair[1])
        print(is_interacting)



if __name__ == "__main__":
    main()
