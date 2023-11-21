from tensorflow.keras.models import load_model
from spektral.layers import GCNConv, GlobalAvgPool
import h5py
import os
import h5py
from scipy.sparse import csr_matrix
from spektral.data import DisjointLoader, Graph, Dataset
import numpy as np
import os
from tqdm import tqdm
import random
from scipy.sparse import block_diag, hstack, vstack

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

def get_model():
    return load_model("graph_ml/checkpoint_iPPI_model.h5", custom_objects={'GCNConv': GCNConv, 'GlobalAvgPool': GlobalAvgPool})

def get_combined_graph(uniprot_A, uniprot_B, smiles):
    prot_A_graph_filename = 'data/protein_atom_graphs/AF-{}-F1-model_v4_graph.hdf5'.format(uniprot_A)
    prot_B_graph_filename = 'data/protein_atom_graphs/AF-{}-F1-model_v4_graph.hdf5'.format(uniprot_B)
    mol_graph_filename = 'data/mol_graphs/{}_graph.hdf5'.format(smiles)
    if not os.path.exists(prot_A_graph_filename) or not os.path.exists(prot_B_graph_filename) or not os.path.exists(mol_graph_filename):
        """ Often the AF is the one not in the data.
        if not os.path.exists(prot_A_graph_filename):
            print(prot_A_graph_filename + " did not exist")
        if not os.path.exists(prot_B_graph_filename):
            print(prot_B_graph_filename + " did not exist")
        if not os.path.exists(mol_graph_filename):
            print(mol_graph_filename + " did not exist")
        """
        return None
    
    csr_protA, feat_protA = load_graph(prot_A_graph_filename)
    csr_protB, feat_protB = load_graph(prot_B_graph_filename)
    csr_mol, feat_mol = load_graph(mol_graph_filename)

    # Combine the adjacency matrices
    combined_adjacency = block_diag((csr_protA, csr_protB, csr_mol))

    # Combine the feature matrices
    combined_features = np.vstack((feat_protA, feat_protB, feat_mol))

    # Create a Spektral Graph object
    combined_graph = Graph(x=combined_features, a=combined_adjacency, y=-1)

    return combined_graph

class PredDataset(Dataset):
    def __init__(self, graphs, **kwargs):
        self.graphs = graphs
        super().__init__(**kwargs)

    def read(self):
        return self.graphs

def load_graph_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        # Load the features matrix
        features = f['features'][:]

        # Load the CSR components and reconstruct the adjacency matrix
        data = f['data'][:]
        indices = f['indices'][:]
        indptr = f['indptr'][:]
        shape = f['shape'][:]
        adjacency = csr_matrix((data, indices, indptr), shape=shape)

        # Load the labels or targets if they exist
        label = f['labels'][:] if 'labels' in f else None

    # Create the Spektral Graph object
    graph = Graph(x=features, a=adjacency, y=label)

    return graph

def predict_from_uniprots_and_smiles(uniprot_A, uniprot_B, smiles, model):
    combined_graph = get_combined_graph(uniprot_A, uniprot_B, smiles)
    if combined_graph is not None:
        return predict([combined_graph], model)
    else:
        return None

def predict(graphs, model):    
    dataset = PredDataset(graphs=graphs)
    loader = DisjointLoader(dataset, batch_size=1, epochs=1)

    y = model.predict(loader.load(), verbose=0)
    return y

def main():
    graphs = []
    labels = []
    file_name = "data/iPPI_graphs/test_graphs/"
    for file in random.sample(sorted(list(os.listdir(file_name))), len(sorted(list(os.listdir(file_name))))):
        graph = load_graph_from_hdf5(os.path.join(file_name, file))
        label = int(file.split("_iPPI_")[1][0])
        labels.append(label)
        graphs.append(graph)

    print(np.array(labels).mean())
    
    """
    combined = list(zip(y, labels))
    for x in combined:
        print(x)
    """

if __name__ == "__main__":
    main()