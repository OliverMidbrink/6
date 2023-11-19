import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from spektral.layers import GCNConv, GlobalAvgPool

def create_gnn_model(n_features, n_classes):
    # Inputs
    node_input = Input(shape=(None, n_features), name='node_input')
    adj_input = Input(shape=(None, None), dtype=tf.float32, sparse=True, name='adj_input')
    # For Disjoint mode, you also need a segment_ids input that tells which nodes belong to which graph
    segment_ids = Input(shape=(None,), dtype=tf.int32, name='segment_ids')

    # GCN layers
    # Define the dropout rate
    dropout_rate = 0.5

    gc1 = GCNConv(256, activation='relu')([node_input, adj_input])
    gc1 = Dropout(dropout_rate)(gc1)
    gc1 = GCNConv(256, activation='relu')([gc1, adj_input])
    gc1 = Dropout(dropout_rate)(gc1)
    gc1 = GCNConv(256, activation='relu')([gc1, adj_input])
    gc1 = Dropout(dropout_rate)(gc1)

    # A global pooling layer to combine node features into graph features
    # The segment_ids tensor is used to perform this pooling operation
    pool = GlobalAvgPool()([gc1, segment_ids])

    # Output layer
    output = Dense(n_classes, activation='softmax')(pool)

    # Create the model
    model = Model(inputs=[node_input, adj_input, segment_ids], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
