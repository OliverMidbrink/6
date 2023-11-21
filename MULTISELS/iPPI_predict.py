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


def get_combined_graph_from_uniprot_and_smiles():
    pass


def predict(graphs):
    model = load_model("graph_ml/checkpoint_iPPI_model.h5", custom_objects={'GCNConv': GCNConv, 'GlobalAvgPool': GlobalAvgPool})
    
    dataset = PredDataset(graphs=graphs)
    loader = DisjointLoader(dataset, batch_size=1, epochs=1)

    y = model.predict(loader.load())

    return y


def main():
    graphs = []
    labels = []
    file_name = "data/iPPI_graphs/train_graphs/"
    for file in sorted(list(os.listdir(file_name)))[:100]:
        graph = load_graph_from_hdf5(os.path.join(file_name, file))
        label = file.split("_iPPI_")[1][0]
        labels.append(label)
        print(label)
        graphs.append(graph)


    y = [float(x[0]) for x in predict(graphs)]
    combined = list(zip(y, labels))
    for x in combined:
        print(x)

if __name__ == "__main__":
    main()