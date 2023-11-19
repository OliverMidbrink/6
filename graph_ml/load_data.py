# load_data.py
import os
import numpy as np
import h5py
from scipy.sparse import csr_matrix
from spektral.data import Graph, Dataset
from tqdm import tqdm 
import json
import random

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

class ProteinGraphDataset(Dataset):
    def __init__(self, graph_data_dir_path, alphabetic_id_one_hot_data_dir_path, uniprot_ids, **kwargs):
        self.graph_data_dir_path = graph_data_dir_path
        self.alphabetic_id_one_hot_data_dir_path = alphabetic_id_one_hot_data_dir_path
        self.uniprot_ids = uniprot_ids
        super().__init__(**kwargs)

    def get_filenames(self, uniprot_id_list):
        return ["AF-{}-F1-model_v4_graph.hdf5".format(x) for x in uniprot_id_list]
    
    def read(self):
        graphs = []
        for file_name in tqdm(self.get_filenames(self.uniprot_ids), desc='Loading graphs', unit='graph'):
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