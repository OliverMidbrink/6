from protein_interaction_predictor_model import *
from model import *
import os
import tensorflow as tf
from spektral.layers import GCNConv, GlobalAvgPool

def get_n_uniprot_ids(af_folder="data/AlphaFoldData/"):
    uniprot_ids = {x.split("-")[1] for x in os.listdir(af_folder) if "AF-" in x}
    sorted_ids = sorted(uniprot_ids)
    print(len(sorted_ids))
    return len(sorted_ids)


def main():
    # Load the pre-trained model
    pretrained_model = tf.keras.models.load_model('graph_ml/20_proteins_classification_model.keras', custom_objects={'GCNConv': GCNConv, 'GlobalAvgPool': GlobalAvgPool})

    # Create the new model
    model = create_protein_interaction_predictor_model(4)

    # Transfer weights -- here's a simplified version assuming the new model
    # has parts of its architecture identical to the pretrained model
    for layer in model.layers:
        print("Layer name: {}".format(layer.name))
        if 'GCNConv' in layer.name:
            layer.set_weights(pretrained_model.get_layer(name=layer.name).get_weights())
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()


if __name__ == "__main__":
    main()