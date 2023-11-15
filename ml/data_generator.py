from data_functions import get_4_channel_grid_data_from_smiles, get_4_channel_grid_data_from_uniprot
import json
import random
import numpy as np
import numpy as np

def get_data_item(synthetic_iPPI_data_json_item):
    # Assuming the data item needs to be processed for grid data
    smiles_data = synthetic_iPPI_data_json_item['SMILES']
    proteins = synthetic_iPPI_data_json_item['proteins']
    
    grid_data = [get_4_channel_grid_data_from_smiles(smiles) for smiles in smiles_data]
    grid_data.extend(get_4_channel_grid_data_from_uniprot(protein) for protein in proteins)

    # Convert the list of grid data to a NumPy array
    return np.array(grid_data)

def load_synthetic_iPPI_data():
    try:
        with open("../DLiP_rule_of_5_compound_data.json", "r") as file:
            return json.load(file)
    except Exception as e:
        print(e)
        print("Please try installing the DLiP_iPPI_data with the download_DLiP scripts in the root folder.")
        return None

def check_if_pair_in_DLiP(synthetic_iPPI_data, uniprot_A, uniprot_B):
    for item in synthetic_iPPI_data.values():
        if set([uniprot_A, uniprot_B]) <= set(item['proteins']):
            return True
    return False

def random_uniprot_from_DLiP(synthetic_iPPI_data):
    random_item = random.choice(list(synthetic_iPPI_data.values()))
    return random.choice(random_item['proteins'])

def get_random_smiles(synthetic_iPPI_data):
    random_key = random.choice(list(synthetic_iPPI_data.keys()))
    return random.choice(synthetic_iPPI_data[random_key]['SMILES'])

def get_random_X_Y_pair(synthetic_iPPI_data):
    # Retrieve random data
    random_uniprot_A = random_uniprot_from_DLiP(synthetic_iPPI_data)
    random_uniprot_B = random_uniprot_from_DLiP(synthetic_iPPI_data)
    random_mol_smiles = get_random_smiles(synthetic_iPPI_data)

    # Convert to suitable format for the model
    random_X_A = get_4_channel_grid_data_from_uniprot(random_uniprot_A)  # shape should be (4, 100, 100, 100)
    random_X_B = get_4_channel_grid_data_from_uniprot(random_uniprot_B)  # shape should be (4, 100, 100, 100)
    random_X_Mol = get_4_channel_grid_data_from_smiles(random_mol_smiles)  # shape should be (8, 30, 30, 30)

    # Check interaction and inhibition
    Y_A_B_interacts_true_false = check_if_pair_in_DLiP(synthetic_iPPI_data, random_uniprot_A, random_uniprot_B)
    Y_mol_is_inhibitor_true_false = Y_A_B_interacts_true_false  # Assuming this is correct

    # Convert to numerical format (1 for True, 0 for False)
    Y_interaction = 1 if Y_A_B_interacts_true_false else 0
    Y_inhibition = 1 if Y_mol_is_inhibitor_true_false else 0

    # Prepare inputs and outputs in the format expected by the model
    X = [np.array(random_X_A), np.array(random_X_B), np.array(random_X_Mol)]
    Y = np.array([Y_interaction, Y_inhibition])

    return X, Y


def random_slice(slice_fraction, data):
    # Shuffle the data
    np.random.shuffle(data)

    # Calculate slice size
    slice_size = int(len(data) * slice_fraction)

    # Return the sliced data
    return data[:slice_size]


np.random.seed(42)  # Set a random seed for reproducibility (optional)

full_synthetic_iPPI_data = load_synthetic_iPPI_data()

# Ensure the data is in a format suitable for slicing (e.g., list or array)
full_synthetic_iPPI_data = list(full_synthetic_iPPI_data)

# Calculate fractions for splitting
total_size = len(full_synthetic_iPPI_data)
train_fraction = 0.7 * 0.75
val_fraction = 0.3 * 0.75
test_fraction = 0.25

# Shuffle and slice data
shuffled_data = np.random.permutation(full_synthetic_iPPI_data)
train_data = random_slice(train_fraction, shuffled_data)
val_data = random_slice(val_fraction, shuffled_data[len(train_data):])
test_data = random_slice(test_fraction, shuffled_data[len(train_data) + len(val_data):])

def train_batch_generator(train_data_synthetic_iPPL):


    return X and Y in good format

def val_batch_generator(val_data_synthetic_iPPL):


    return X and Y in good format


def test_batch_generator(test_data_synthetic_iPPL):


    return X and Y in good format

def main():
    print("adfasdf")

if __name__ == "__main__":
    main()
