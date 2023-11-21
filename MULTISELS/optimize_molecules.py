import json
import random
import os
import h5py
from scipy.sparse import csr_matrix
from spektral.data import Graph
import numpy as np
from iPPI_predict import get_model, predict_from_uniprots_and_smiles
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    tuples = get_targets_iPPIs("MULTISELS/OSK_upreg_2_neighbors_chatGPT3_turpo_1106.json")
    
    if os.path.exists("MULTISELS/smile_scores.json"):
        smile_scores = []
        with open("MULTISELS/smile_scores.json", "r") as file:
            smile_scores = json.load(file).values()


        plt.plot([data["total_score"] for data in smile_scores])
        plt.show()
        return None

    model = get_model()
    smiles_subset = random.sample(get_smiles(), 100)

    smile_scores = {}
    id = 0
    for smile in tqdm(smiles_subset, desc="Getting smile scores", unit="smiles"):
        smile_score = 0
        for tuple in tuples:
            iPPI_prob = predict_from_uniprots_and_smiles(tuple[0], tuple[1], smile, model)
            if iPPI_prob is not None:
                iPPI_prob = iPPI_prob[0][0]
                #print(iPPI_prob)
            
                smile_score += iPPI_prob * tuple[2]
                #print("Added {} to score.".format(iPPI_prob * tuple[2]))
            else:
                print("Could not analyze")
                #smile_score += 0.5 * tuple[2]
                #print("Assumed 0.5 iPPI_prob and added {} to score.".format(0.5 * tuple[2]))
        
        smile_scores[id] = {"smile": smile, "total_score": smile_score}
        id += 1

    with open("MULTISELS/smile_scores.json", "w") as file:
        json.dump(smile_scores, file)
    

if __name__ == "__main__":
    main()