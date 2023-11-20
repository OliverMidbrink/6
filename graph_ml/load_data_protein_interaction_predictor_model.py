# load_data.py
import os
import numpy as np
import h5py
from scipy.sparse import csr_matrix
from spektral.data import Graph, Dataset
from tqdm import tqdm 
import json
import random
from interactome_tools import *
from scipy.sparse import block_diag

def get_uniprot_ids(af_folder="data/AlphaFoldData/"):
    uniprot_ids = {x.split("-")[1] for x in os.listdir(af_folder) if "AF-" in x}
    sorted_ids = sorted(uniprot_ids)
    print(len(sorted_ids))
    return sorted_ids

def get_train_val_test_split():
    train_uniprot_ids = []
    val_uniprot_ids = []
    test_uniprot_ids = [] 
    if os.path.exists("graph_ml/train_uniprot_ids.json") and os.path.exists("graph_ml/val_uniprot_ids.json") and os.path.exists("graph_ml/test_uniprot_ids.json"):
        print("Loading existing uniprot id lists for train val test")
        with open("graph_ml/train_uniprot_ids.json", "r") as file:
            train_uniprot_ids = json.load(file)["uniprot_ids_train"]
        with open("graph_ml/val_uniprot_ids.json", "r") as file:
            val_uniprot_ids = json.load(file)["uniprot_ids_val"]
        with open("graph_ml/test_uniprot_ids.json", "r") as file:
            test_uniprot_ids = json.load(file)["uniprot_ids_test"]
    else:
        print("Creating uniprot id lists for train val test")
        shuffled_uniprot_id_list = get_uniprot_ids()
        random.shuffle(shuffled_uniprot_id_list)

        split_train_val = int(0.8 * len(shuffled_uniprot_id_list))
        split_val_test = int(0.9 * len(shuffled_uniprot_id_list))

        # Split the shuffled list into training, validation, and testing sets
        train_uniprot_ids = shuffled_uniprot_id_list[:split_train_val]
        val_uniprot_ids = shuffled_uniprot_id_list[split_train_val:split_val_test]
        test_uniprot_ids = shuffled_uniprot_id_list[split_val_test:]


        with open("graph_ml/train_uniprot_ids.json", "w") as file:
            train_dict = {"uniprot_ids_train": train_uniprot_ids}
            json.dump(train_dict, file)
        with open("graph_ml/val_uniprot_ids.json", "w") as file:
            val_dict = {"uniprot_ids_val": val_uniprot_ids}
            json.dump(val_dict, file)
        with open("graph_ml/test_uniprot_ids.json", "w") as file:
            test_dict = {"uniprot_ids_test": test_uniprot_ids}
            json.dump(test_dict, file)

    return train_uniprot_ids, val_uniprot_ids, test_uniprot_ids

class ProteinPairLabelDataset(Dataset):
    def __init__(self, graph_data_dir_path, alphabetic_id_one_hot_data_dir_path, uniprot_ids, full_uniprot_ids, uniprot_id_pairs_file_path, sample=None, **kwargs):
        self.graph_data_dir_path = graph_data_dir_path
        self.alphabetic_id_one_hot_data_dir_path = alphabetic_id_one_hot_data_dir_path
        self.uniprot_ids = uniprot_ids
        self.full_uniprot_ids = full_uniprot_ids
        self.uniprot_id_pairs_file_path = uniprot_id_pairs_file_path
        self.sample = sample
        super().__init__(**kwargs)

    def get_filenames(self, uniprot_id_list):
        return ["AF-{}-F1-model_v4_graph.hdf5".format(x) for x in uniprot_id_list]
    
    def get_protein_pairs_and_labels(self):
        protein_pairs_and_labels = []

        if os.path.exists(self.uniprot_id_pairs_file_path):
            print("Loading pairs from file.")
            with open(self.uniprot_id_pairs_file_path, "r") as file:
                protein_pairs_and_labels = json.load(file)["protein_pairs_and_labels"]
        else:
            print("Creating protein pairs")
            # Load the HuRI csr matrix outside the loop to avoid reloading it for each protein
            HuRI_csr_matrix = load_HuRI_csr_matrix()
            # Wrap self.uniprot_ids with tqdm to display progress
            for uniprot_id in tqdm(self.uniprot_ids, desc='Generating protein pairs', unit="uniprot_ids"):
                neighbors = get_neighbors(HuRI_csr_matrix, self.full_uniprot_ids, uniprot_id)

                for neighbor_uniprot_id in neighbors:
                    if neighbor_uniprot_id not in self.uniprot_ids:
                        continue

                    # Append interacting pair
                    protein_pairs_and_labels.append((uniprot_id, neighbor_uniprot_id, 1))

                    # Append non-interacting pair
                    non_interact_a, non_interact_b = get_non_interacting_uniprot_ids(HuRI_csr_matrix, self.uniprot_ids)
                    protein_pairs_and_labels.append((non_interact_a, non_interact_b, 0))

            print("Saving protein pairs to {}...".format(self.uniprot_id_pairs_file_path))
            with open(self.uniprot_id_pairs_file_path, "w") as file:
                json.dump({"protein_pairs_and_labels": protein_pairs_and_labels}, file)

        return protein_pairs_and_labels
    
    def combine_spektral_graphs(self, graph1, graph2, label):
        # Combine feature matrices
        combined_features = np.concatenate((graph1.x, graph2.x), axis=0)
        
        # Create a block diagonal adjacency matrix
        combined_adjacency = block_diag((graph1.a, graph2.a))
        
        # Create a combined Spektral Graph object
        # If your graphs include edge features or other attributes, you should combine them here as well.
        combined_graph = Graph(x=combined_features, a=combined_adjacency, y=np.array(label, dtype=np.float32))
        
        return combined_graph

    def read(self):
        # Get the list of protein pairs and interaction labels
        protein_pairs_and_labels = self.get_protein_pairs_and_labels()
        if not os.path.exists("data/combined_graphs"):
            os.makedirs("data/combined_graphs")
        
        graphs = []

        if self.sample is None:
            go_trough = protein_pairs_and_labels
        else:
            go_trough = random.sample(protein_pairs_and_labels, self.sample)
        for protein_id1, protein_id2, interaction_label in tqdm(go_trough, desc='Loading graph pairs', unit='pair'):
            combined_graph_file_path = "data/combined_graphs/{}_{}_{}.hdf5".format(protein_id1, protein_id2, interaction_label)
            if os.path.exists(combined_graph_file_path):
                combined_graph = self.load_combined_graph_from_hdf5(combined_graph_file_path)
                graphs.append(combined_graph)
            else:
                #print("Creating combined graph")
                # Load the graphs for both proteins
                graph1 = self.load_graph_from_hdf5(os.path.join(self.graph_data_dir_path, self.get_filenames([protein_id1])[0]))
                graph2 = self.load_graph_from_hdf5(os.path.join(self.graph_data_dir_path, self.get_filenames([protein_id2])[0]))

                combined_graph = self.combine_spektral_graphs(graph1, graph2, interaction_label)

                graphs.append(combined_graph)

                # Save graph for quicker access next time Do not save because load function does not work
                #self.save_combined_graph_to_hdf5(combined_graph, combined_graph_file_path)
        return graphs
    
    def save_combined_graph_to_hdf5(self, graph, file_path):
        with h5py.File(file_path, 'w') as f:
            # Store features
            f.create_dataset('features', data=graph.x)
            # Store labels
            if graph.y is not None:
                f.create_dataset('y', data=graph.y)
            # Check if the adjacency matrix is a CSR matrix
            if isinstance(graph.a, csr_matrix):
                # Store adjacency in CSR format
                f.create_dataset('data', data=graph.a.data)
                f.create_dataset('indices', data=graph.a.indices)
                f.create_dataset('indptr', data=graph.a.indptr)
                f.attrs['shape'] = graph.a.shape
            else:
                # Convert to CSR and then store
                csr_adj = csr_matrix(graph.a)
                f.create_dataset('data', data=csr_adj.data)
                f.create_dataset('indices', data=csr_adj.indices)
                f.create_dataset('indptr', data=csr_adj.indptr)
                f.attrs['shape'] = csr_adj.shape
    
    def load_combined_graph_from_hdf5(self, file_path):
        with h5py.File(file_path, 'r') as f:
            # Load the features matrix
            features = f['features'][:]
            # Load the labels if they exist
            labels = f['y'] if 'y' in f else None
            # Load the CSR components
            data = f['data'][:]
            indices = f['indices'][:]
            indptr = f['indptr'][:]
            shape = f.attrs['shape']
            # Reconstruct the CSR matrix
            csr_graph = csr_matrix((data, indices, indptr), shape=shape)

        # Create a Spektral Graph object including the labels
        graph = Graph(x=features, a=csr_graph, y=labels)

        return graph


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

        # Create a Spektral Graph object
        graph = Graph(x=features, a=csr_graph)

        return graph

    def load_one_hot_vector(self, file_name):
        # Constructs the full path to the HDF5 file
        file_path = os.path.join(self.alphabetic_id_one_hot_data_dir_path, file_name)
        # Uses the h5py library to load the one-hot vector from the HDF5 file
        with h5py.File(file_path, 'r') as h5f:
            one_hot_vector = h5f['one_hot'][:]
        return one_hot_vector