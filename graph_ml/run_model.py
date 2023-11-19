import tensorflow as tf
from load_data import ProteinGraphDataset, get_train_val_test_split
from model import create_gnn_model
from spektral.data import DisjointLoader

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
    train_uniprot_ids, val_uniprot_ids, test_uniprot_ids = get_train_val_test_split()
    train_dataset = ProteinGraphDataset(graph_data_dir_path="data/protein_atom_graphs", alphabetic_id_one_hot_data_dir_path="data/protein_one_hot_id_vectors", uniprot_ids=train_uniprot_ids)
    val_dataset = ProteinGraphDataset(graph_data_dir_path="data/protein_atom_graphs", alphabetic_id_one_hot_data_dir_path="data/protein_one_hot_id_vectors", uniprot_ids=val_uniprot_ids)
    test_dataset = ProteinGraphDataset(graph_data_dir_path="data/protein_atom_graphs", alphabetic_id_one_hot_data_dir_path="data/protein_one_hot_id_vectors", uniprot_ids=test_uniprot_ids)

    # Estimate the number of classes from the dataset
    n_classes = train_dataset.n_classes

    # Create the model
    model = create_gnn_model(n_features=4, n_classes=n_classes)
    model.summary()

    # Prepare the data loader
    train_loader = DisjointLoader(train_dataset, batch_size=4, epochs=1000)
    val_loader = DisjointLoader(val_dataset, batch_size=4)
    test_loader = DisjointLoader(test_dataset, batch_size=4)
    
    # Train the model
    history = model.fit(train_loader.load(), validation_data=val_loader.load(), steps_per_epoch=train_loader.steps_per_epoch, epochs=1000)

    model.save('graph_ml/proteins_classification_model_start_2023-11-19_1000_epochs.keras')


if __name__ == "__main__":
    main()
