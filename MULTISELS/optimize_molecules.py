import json
import random
import os
import h5py
from scipy.sparse import csr_matrix
from spektral.data import Graph
import numpy as np
from iPPI_predict import get_model, predict_from_uniprots_and_smiles

def get_targets_iPPIs(file_path):
    with open(file_path, "r") as file:
        return json.load(file)["tuples"]

def get_smiles():
    smiles = set()
    for file_name in os.listdir("data/mol_graphs"):
        smile = file_name.split("_graph")[0]
        smiles.add(smile)
    return sorted(list(smiles))

def load_graph(file_path):
    with h5py.File(file_path, 'r') as f:
        # Load the features matrix
        features = f['features'][:]
        # Load the CSR components
        data = f['data'][:]
        indices = f['indices'][:]
        indptr = f['indptr'][:]
        shape = f['shape'][:]
        # Create the CSR matrix
        csr_graph = csr_matrix((data, indices, indptr), shape=shape)
    
    return csr_graph, features

def main():
    model = get_model()
    tuples = get_targets_iPPIs("MULTISELS/OSK_upreg_2_neighbors_chatGPT3_turpo_1106.json")
    
    smiles_subset = random.sample(get_smiles(), 100)

    for smile in smiles_subset:
        for tuple in tuples:
            iPPI_prob = predict_from_uniprots_and_smiles(tuple[0], tuple[1], smile, model)
            if iPPI_prob is not None:
                iPPI_prob = iPPI_prob[0][0]
                print(iPPI_prob)
    for x in tuples:
        pass
if __name__ == "__main__":
    main()