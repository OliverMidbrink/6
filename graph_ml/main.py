import numpy as np
import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from spektral.layers import GraphConv, GlobalAvgPool


def create_gnn_block():
    return keras.Sequential([
        GraphConv(64, activation='relu'),
        GlobalAvgPool()
    ])


def create_model():
    # Protein Branches
    protein_input_1 = Input(shape=(None, None), name='protein_input_1')
    protein_input_2 = Input(shape=(None, None), name='protein_input_2')

    protein_gnn_1 = create_gnn_block()(protein_input_1)
    protein_gnn_2 = create_gnn_block()(protein_input_2)

    combined_protein = concatenate([protein_gnn_1, protein_gnn_2])
    protein_representation = Dense(128, activation='relu')(combined_protein)

    # Molecule Branch
    molecule_input = Input(shape=(None, None), name='molecule_input')
    molecule_gnn = create_gnn_block()(molecule_input)
    molecule_representation = Dense(128, activation='relu')(molecule_gnn)

    # Combined Representation
    combined_representation = concatenate([protein_representation, molecule_representation])
    combined_representation = Dense(128, activation='relu')(combined_representation)

    # Outputs
    ppi_prediction = Dense(1, activation='sigmoid', name='ppi_output')(protein_representation)
    inhibition_prediction = Dense(1, activation='sigmoid', name='inhibition_output')(combined_representation)

    # Create and compile the model
    model = Model(inputs=[protein_input_1, protein_input_2, molecule_input], 
                  outputs=[ppi_prediction, inhibition_prediction])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model



model = create_model()
model.summary()
