from protein_mol_interaction_model import ProteinMolInteractionModel
from data_generator import get_train_val_test_generators, load_synthetic_iPPI_data

def main():
    full_synthetic_iPPI_data = load_synthetic_iPPI_data()
    train_gen, val_gen, test_gen = get_train_val_test_generators(full_synthetic_iPPI_data, 0.7 * 0.75, 0.3 * 0.75, 0.25, 32)

    protein_mol_interaction_model = ProteinMolInteractionModel()
    protein_mol_interaction_model.summary()

    protein_mol_interaction_model.compile(
        optimizer='adam',
        loss=['binary_crossentropy', 'binary_crossentropy'],
        metrics=['accuracy']
    )

    # Recommended values for steps_per_epoch and epochs:
    steps_per_epoch = len(train_gen) // 32  # Adjust batch size if needed
    epochs = 10  # Adjust as needed

    protein_mol_interaction_model.fit(train_gen, val_gen, steps_per_epoch=steps_per_epoch, batch_size=32, epochs=epochs)

    # Evaluate the model on the test data
    test_results = protein_mol_interaction_model.evaluate(test_gen, batch_size=32)

    # Print the evaluation results
    print("Test Loss:", test_results[0])
    print("Test Accuracy for Protein-Protein Interaction:", test_results[1])
    print("Test Accuracy for Molecular Inhibitor:", test_results[2])

if __name__ == "__main__":
    main()