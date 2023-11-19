import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from spektral.layers import GCNConv, GlobalAvgPool

def create_protein_pipe(n_features, name):
    # Inputs
    node_input = Input(shape=(None, n_features), name='node_input')
    adj_input = Input(shape=(None, None), dtype=tf.float32, sparse=True, name='adj_input')
    segment_ids = Input(shape=(None,), dtype=tf.int32, name='segment_ids')

    # GCN layers with dropout
    dropout_rate = 0.5
    gc1 = GCNConv(256, activation='relu')([node_input, adj_input])
    gc1 = Dropout(dropout_rate)(gc1)
    gc1 = GCNConv(256, activation='relu')([gc1, adj_input])
    gc1 = Dropout(dropout_rate)(gc1)
    gc1 = GCNConv(256, activation='relu')([gc1, adj_input])
    gc1 = Dropout(dropout_rate)(gc1)

    # Global pooling
    pool = GlobalAvgPool()([gc1, segment_ids])

    return Model(inputs=[node_input, adj_input, segment_ids], outputs=pool, name=name)

def create_protein_interaction_predictor_model(n_features):
    # Create two protein pipes
    protein_pipe1 = create_protein_pipe(n_features, "ProteinPipe1")
    protein_pipe2 = create_protein_pipe(n_features, "ProteinPipe2")

    # Define inputs for each protein pipe
    input_a = [Input(shape=(None, n_features), name='node_input_a'),
               Input(shape=(None, None), dtype=tf.float32, sparse=True, name='adj_input_a'),
               Input(shape=(None,), dtype=tf.int32, name='segment_ids_a')]
    input_b = [Input(shape=(None, n_features), name='node_input_b'),
               Input(shape=(None, None), dtype=tf.float32, sparse=True, name='adj_input_b'),
               Input(shape=(None,), dtype=tf.int32, name='segment_ids_b')]

    # Get outputs from both protein pipes
    output_a = protein_pipe1(input_a)
    output_b = protein_pipe2(input_b)

    # Concatenate outputs
    merged = concatenate([output_a, output_b])

    # Prediction output
    output = Dense(1, activation='sigmoid')(merged)

    # Create the final model
    model = Model(inputs=input_a + input_b, outputs=output)

    return model

