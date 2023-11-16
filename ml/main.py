from protein_mol_interaction_model import ProteinMolInteractionModel
from data_generator import get_train_val_test_generators, load_synthetic_iPPI_data
import tensorflow as tf
import matplotlib as plt
from tensorflow.keras.metrics import BinaryAccuracy


def accuracy_for_label_1(y_true, y_pred):
    # Select the predictions and true values for label 1
    y_pred_label_1 = y_pred[:, 0]
    y_true_label_1 = y_true[:, 0]
    correct_predictions = tf.equal(tf.round(y_pred_label_1), tf.round(y_true_label_1))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def accuracy_for_label_2(y_true, y_pred):
    # Select the predictions and true values for label 2
    y_pred_label_2 = y_pred[:, 1]
    y_true_label_2 = y_true[:, 1]
    correct_predictions = tf.equal(tf.round(y_pred_label_2), tf.round(y_true_label_2))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))



def main():
    batch_size = 10
    full_synthetic_iPPI_data = load_synthetic_iPPI_data()
    train_gen, val_gen, test_gen, train_len, val_len, test_len = get_train_val_test_generators(batch_size=batch_size)

    protein_mol_interaction_model = ProteinMolInteractionModel()
    protein_mol_interaction_model.summary()

    protein_mol_interaction_model.compile(
        optimizer='adam',
        loss=['binary_crossentropy', 'binary_crossentropy'],
        metrics=[accuracy_for_label_1, accuracy_for_label_2]
    )


    # Recommended values for steps_per_epoch and epochs:
    steps_per_epoch = train_len // batch_size  # Adjust batch size if needed
    val_steps = val_len // batch_size
    test_steps = test_len // batch_size
    epochs = 3  # Adjust as needed

    # TODO add validation generator
    history = protein_mol_interaction_model.fit(train_gen, validation_data=val_gen, steps_per_epoch=steps_per_epoch, 
                                                batch_size=batch_size, validation_batch_size=batch_size,
                                                validation_steps=val_steps, epochs=epochs)
    
    tf.saved_model.save(protein_mol_interaction_model, 'protein_mol_interaction_model')


    # Evaluate the model on the test data
    test_results = protein_mol_interaction_model.evaluate(test_gen, steps=test_steps, batch_size=batch_size)
    print("Test Loss:", test_results[0])
    print("Test Accuracy for Label 1:", test_results[1])
    print("Test Accuracy for Label 2:", test_results[2])

    # Plotting Training Accuracy for each output
    plt.figure(figsize=(12, 5))

    # Output 1 Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['output1_binary_accuracy'])  # Adjust the key based on your output layer name
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy for Output 1')
    plt.title('Training Accuracy Over Time for Output 1')

    # Output 2 Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['output2_binary_accuracy'])  # Adjust the key based on your output layer name
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy for Output 2')
    plt.title('Training Accuracy Over Time for Output 2')

    plt.show()


if __name__ == "__main__":
    main()