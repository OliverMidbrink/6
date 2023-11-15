from protein_mol_interaction_model import ProteinMolInteractionModel
from data_generator import get_train_val_test_generators, load_synthetic_iPPI_data

def main():
    batch_size = 2
    full_synthetic_iPPI_data = load_synthetic_iPPI_data()
    train_gen, val_gen, test_gen, train_len, val_len, test_len = get_train_val_test_generators(batch_size=batch_size)

    protein_mol_interaction_model = ProteinMolInteractionModel()
    protein_mol_interaction_model.summary()

    protein_mol_interaction_model.compile(
        optimizer='adam',
        loss=['binary_crossentropy', 'binary_crossentropy'],
        metrics=['accuracy']
    )

    # Recommended values for steps_per_epoch and epochs:
    steps_per_epoch = train_len // batch_size  # Adjust batch size if needed
    epochs = 2  # Adjust as needed

    history = protein_mol_interaction_model.fit(train_gen, validation_data=val_gen, steps_per_epoch=steps_per_epoch, batch_size=batch_size, epochs=epochs)

    # Evaluate the model on the test data
    test_results = protein_mol_interaction_model.evaluate(test_gen, batch_size=batch_size)

    # Print the evaluation results
    print("Test Loss:", test_results[0])
    print("Test Accuracy for Protein-Protein Interaction:", test_results[1])
    print("Test Accuracy for Molecular Inhibitor:", test_results[2])

    plt.plot(history.history['accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy Over Time')
    plt.show()

if __name__ == "__main__":
    main()