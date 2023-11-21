import h5py
from scipy.sparse import csr_matrix
from spektral.data import Graph, Dataset
from tensorflow.keras.utils import Sequence
import numpy as np
import os
from tqdm import tqdm

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
        label = f['labels'] if 'labels' in f else None

    # Create the Spektral Graph object
    graph = Graph(x=features, a=adjacency, y=label)

    return graph


class iPPIDataset(Dataset):
    def __init__(self, mode, **kwargs):
        self.mode = mode # mode is "train", "val" or "test"
        if mode not in ["train", "val", "test"]:
            print('mode has to be one of "train", "val" or "test"')
            raise ValueError

        super().__init__(**kwargs)


    def read(self):
        graphs = []

        graph_dataset_file = ""
        if self.mode == "train":
            graph_dataset_file = "data/iPPI_graphs/train_graphs"
        if self.mode == "val":
            graph_dataset_file = "data/iPPI_graphs/val_graphs"
        if self.mode == "test":
            graph_dataset_file = "data/iPPI_graphs/test_graphs"


        graph_files = sorted(os.listdir(graph_dataset_file))

        for graph_file in tqdm(graph_files, desc="Loading {} files.".format(self.mode), unit="graphs"):
            graph = load_graph_from_hdf5(os.path.join(graph_dataset_file, graph_file))
            graphs.append(graph)

        return graphs


def train_generator():
    train_data_path = "data/iPPI_graphs/train_graphs"
    train_files = sorted(os.listdir(train_data_path))

    for train_file in train_files:
        graph = load_graph_from_hdf5(os.path.join(train_data_path, train_file))

        yield [graph]