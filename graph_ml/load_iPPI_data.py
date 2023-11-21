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
        label = f['labels'][:] if 'labels' in f else None

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

        for graph_file in tqdm(graph_files[:], desc="Loading {} files.".format(self.mode), unit="graphs"):
            file_name = os.path.join(graph_dataset_file, graph_file)
            graph = load_graph_from_hdf5(file_name)
            graphs.append(graph)

        return graphs


def main():
    file_name = "data/iPPI_graphs/train_graphs/"
    file_name = os.path.join(file_name, os.listdir(file_name)[0])
    graph = load_graph_from_hdf5(file_name)
    print(graph)

if __name__ == "__main__":
    main()