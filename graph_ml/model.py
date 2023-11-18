import spektral
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from spektral.layers import GCNConv

def create_gnn_model(num_features=4, num_classes=23391):
    # Define the model
    node_input = Input(shape=(num_features,), name='node_input')
    adj_input = Input((None,), dtype=tf.float32, sparse=True, name='adj_input')

    # Graph Convolutional layers
    x = GCNConv(64, activation='relu')([node_input, adj_input])
    x = Dropout(0.5)(x)
    x = GCNConv(64, activation='relu')([x, adj_input])
    x = Dropout(0.5)(x)

    # Global pooling can be applied here, if needed

    # Output layer
    output = Dense(num_classes, activation='softmax')(x)  # Use softmax for multi-class classification

    # Create the model
    model = Model(inputs=[node_input, adj_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def main():
    model = create_gnn_model()
    model.summary()


if __name__ == "__main__":
    main()
