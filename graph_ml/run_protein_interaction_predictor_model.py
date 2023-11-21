from protein_interaction_predictor_model import *
from model import *
import os
import tensorflow as tf
from spektral.layers import GCNConv, GlobalAvgPool
from load_data_protein_interaction_predictor_model import ProteinPairLabelDataset, get_train_val_test_split
from spektral.data import DisjointLoader
from tensorflow.keras.callbacks import ModelCheckpoint

def get_uniprot_ids(af_folder="data/AlphaFoldData/"):
    uniprot_ids = {x.split("-")[1] for x in os.listdir(af_folder) if "AF-" in x}
    sorted_ids = sorted(uniprot_ids)
    print(len(sorted_ids))
    return sorted_ids


def main():
    # Load the pre-trained model
    pretrained_model = tf.keras.models.load_model('graph_ml/20_proteins_classification_model.keras', custom_objects={'GCNConv': GCNConv, 'GlobalAvgPool': GlobalAvgPool})

    # Create the new model
    model = create_protein_interaction_predictor_model(9)

    
    # Transfer weights from pretrained model to the protein pipes of "model"
    for model_layer in model.layers:
        for pretrained_model_layer in pretrained_model.layers:
            if pretrained_model_layer.name == model_layer.name and 'gcn_conv' in model_layer.name:
                    print("Setting pretrained weights on {} from {}.".format(model_layer.name, pretrained_model_layer.name))
                    model_layer.set_weights(pretrained_model.get_layer(name=pretrained_model_layer.name).get_weights())
    
    

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()


    full_uniprot_ids = get_uniprot_ids()
    train_uniprot_ids, val_uniprot_ids, test_uniprot_ids = get_train_val_test_split()
    train_dataset = ProteinPairLabelDataset(graph_data_dir_path="data/protein_atom_graphs", alphabetic_id_one_hot_data_dir_path="data/protein_one_hot_id_vectors", uniprot_ids=train_uniprot_ids, full_uniprot_ids=full_uniprot_ids, uniprot_id_pairs_file_path="graph_ml/train_uniprot_pairs.json", sample=40)
    val_dataset = ProteinPairLabelDataset(graph_data_dir_path="data/protein_atom_graphs", alphabetic_id_one_hot_data_dir_path="data/protein_one_hot_id_vectors", uniprot_ids=val_uniprot_ids, full_uniprot_ids=full_uniprot_ids, uniprot_id_pairs_file_path="graph_ml/val_uniprot_pairs.json", sample = 40)
    #test_dataset = ProteinPairLabelDataset(graph_data_dir_path="data/protein_atom_graphs", alphabetic_id_one_hot_data_dir_path="data/protein_one_hot_id_vectors", uniprot_ids=test_uniprot_ids, full_uniprot_ids=full_uniprot_ids, uniprot_id_pairs_file_path="graph_ml/test_uniprot_pairs.json")

    train_loader = DisjointLoader(train_dataset, batch_size=4, epochs=10000000)
    val_loader = DisjointLoader(val_dataset, batch_size=4)
    #test_loader = DisjointLoader(test_dataset, batch_size=4)

    checkpoint_path = "graph_ml/best_protein_interaction_40_train_samples_model.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    history = model.fit(
        train_loader.load(),
        validation_data=val_loader.load(),
        validation_steps=val_loader.steps_per_epoch,
        steps_per_epoch=train_loader.steps_per_epoch,
        epochs=10000000,
        callbacks=[checkpoint]  # Add the checkpoint callback here
    )

    model.save('graph_ml/proteins_interaction_predictor_model_start_2023-11-19_1000_epochs.keras')

if __name__ == "__main__":
    main()