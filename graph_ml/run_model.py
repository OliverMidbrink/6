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

    # Train the model
    history = model.fit(x_train, y_train, epochs=1, batch_size=1)

if __name__ == "__main__":
    main()