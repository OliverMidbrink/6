import tensorflow as tf
from load_data import ProteinGraphDataset
from model import create_gnn_model
from spektral.data import DisjointLoader
import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def main():
    print("loading data")
    dataset = ProteinGraphDataset(graph_data_dir_path="data/protein_atom_graphs", alphabetic_id_one_hot_data_dir_path="data/protein_one_hot_id_vectors")

    # Estimate the number of classes from the dataset
    n_classes = dataset.n_classes

    # Create the model
    model = create_gnn_model(n_features=4, n_classes=n_classes)
    model.summary()

    # Prepare the data loader
    loader = DisjointLoader(dataset, batch_size=4, epochs=100000)
    
    # Train the model
    history = model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=100000)

if __name__ == "__main__":
    main()
