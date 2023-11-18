from model import create_gnn_model
from load_data import load_data
import tensorflow as tf
import numpy as np

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
    model = create_gnn_model()
    model.summary()

    x_train, y_train = load_data()

    max_num_nodes = 0

    # Assuming x_train contains your input data as a list of NumPy arrays
    for data in x_train:
        num_nodes = data.shape[0]
        if num_nodes > max_num_nodes:
            max_num_nodes = num_nodes

    x_train_padded = []
    for data in x_train:
        num_nodes = data.shape[0]
        padded_data = np.zeros((max_num_nodes, num_features))
        padded_data[:num_nodes, :] = data
        x_train_padded.append(padded_data)

    # Convert sparse adjacency matrices to dense matrices and reshape them
    adj_input_dense = []
    for adj_matrix in adj_input:
        dense_matrix = adj_matrix.toarray()  # Convert sparse matrix to dense
        num_nodes = dense_matrix.shape[0]
        padded_matrix = np.zeros((max_num_nodes, max_num_nodes))
        padded_matrix[:num_nodes, :num_nodes] = dense_matrix
        adj_input_dense.append(padded_matrix)

    # Convert the lists to NumPy arrays
    x_train_padded = np.array(x_train_padded)
    adj_input_dense = np.array(adj_input_dense)

    # Train the model
    history = model.fit([x_train_padded, adj_input_dense], y_train, epochs=1, batch_size=1)

if __name__ == "__main__":
    main()