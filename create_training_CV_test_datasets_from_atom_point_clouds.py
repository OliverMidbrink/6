import os, json
import glob
import h5py
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import numpy as np
import random
import h5py
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def group_files_by_uniprot(file_paths):
    grouped_files = defaultdict(list)
    for file_path in file_paths:
        uniprot_id = os.path.basename(file_path).split('-')[1]
        grouped_files[uniprot_id].append(file_path)
    return list(grouped_files.values())



def split_data(grouped_files, test_size=0.25):
    train_groups, test_groups = train_test_split(grouped_files, test_size=test_size, random_state=42)
    return train_groups, test_groups

def create_cv_sets(grouped_files, n_splits=4):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_sets = [(train_idx, val_idx) for train_idx, val_idx in kf.split(grouped_files)]
    return cv_sets



def center_of_mass(coordinates):
    # Assuming each atom has equal weight
    return np.mean(coordinates[:, 1:], axis=0)

def random_rotation():
    # Generate a random rotation matrix
    rotation = R.random()
    return rotation.as_matrix()

def apply_transformations(coordinates, grid_size):
    # Calculate the center of mass
    com = center_of_mass(coordinates)

    # Translate the coordinates to origin
    translation = -com
    coordinates[:, 1:] += translation

    # Apply a random rotation
    rotation_matrix = random_rotation()
    coordinates[:, 1:] = np.dot(coordinates[:, 1:], rotation_matrix)

    # Translate back to center of grid
    translation = np.array(grid_size) / 2 - com
    coordinates[:, 1:] += translation

    # Apply a random translation
    half_grid_size = np.array(grid_size) / 2
    translation = np.random.uniform(-half_grid_size, half_grid_size)
    coordinates[:, 1:] += translation

    # Normalize and scale coordinates to fit in the grid
    # Ensure coordinates are within the grid bounds
    coordinates[:, 1:] = np.clip(coordinates[:, 1:], 0, np.array(grid_size) - 1)
    return coordinates

def map_to_grid(coordinates, grid_size, num_channels):
    volume = np.zeros(grid_size + (num_channels,), dtype=np.float32)
    atom_type_to_channel = {1: 0, 2: 1, 3: 2, 4: 3}

    for coord in coordinates:
        atom_type, x, y, z = coord
        channel_idx = atom_type_to_channel.get(atom_type)
        if channel_idx is not None:
            # Convert to integer grid coordinates
            x, y, z = map(int, [x, y, z])
            volume[x, y, z, channel_idx] = 1

    return volume

def data_generator(file_paths, grid_size=(100, 100, 100), num_channels=4):
    for file_path in file_paths:
        coordinates = read_hdf5(file_path)
        transformed_coordinates = apply_transformations(coordinates, grid_size)
        volume = map_to_grid(transformed_coordinates, grid_size, num_channels)
        yield volume


def build_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model



def train_model(model, train_generator, val_generator, epochs=10, steps_per_epoch=None, validation_steps=None):
    history = model.fit(train_generator, 
                        epochs=epochs, 
                        steps_per_epoch=steps_per_epoch, 
                        validation_data=val_generator,
                        validation_steps=validation_steps)
    return history

def evaluate_model(model, test_generator):
    # Assuming test_data is formatted as (inputs, targets)
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")



def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as h5file:
        return np.array(h5file['atom_coordinates'])

def get_4channel_cloud_data_from_uniprot(uniprot_id, protein_atom_point_clouds_folder=os.path.join("data", "protein_atom_point_clouds"), grid_size=(100, 100, 100), num_channels=4):
    # Only use the first fold of each uniprot
    protein_atom_point_cloud_filename = os.path.join(protein_atom_point_clouds_folder, "AF-{}-F1-model_v4_atom_cloud.hdf5".format(uniprot_id))
    
    coordinates = read_hdf5(protein_atom_point_cloud_filename)
    
    transformed_coordinates = apply_transformations(coordinates, grid_size)

    channels = np.zeros((num_channels, *grid_size))

    # Project coordinates into channels
    for coord in transformed_coordinates:
        atom_type = int(coord[0])  # Assuming the atom type is the first element
        if 1 <= atom_type <= num_channels:
            x, y, z = map(int, coord[1:])
            channels[atom_type - 1, x, y, z] = 1  # Mark the presence of the atom

    return channels


def calculate_max_axis_from_uniprot(protein_atom_point_clouds_folder, uniprot_id):
    # Only use the first fold of each uniprot
    protein_atom_point_cloud_filename = os.path.join(protein_atom_point_clouds_folder, "AF-{}-F1-model_v4_atom_cloud.hdf5".format(uniprot_id))
    
    np_array = read_hdf5(protein_atom_point_cloud_filename)
    
    last_three_columns = np_array[:, 1:]

    max_values = np.max(last_three_columns, axis=0)
    min_values = np.min(last_three_columns, axis=0)

    furthest_point_from_zero = np.maximum(np.abs(max_values), np.abs(min_values))

    return furthest_point_from_zero

def shuffle_and_split_uniprot_ids_simple(input_prot_id_list):
    prot_ids_list_shuffled = input_prot_id_list.copy()
    random.shuffle(prot_ids_list_shuffled)

    split_test_idx = int(round(0.25 * len(prot_ids_list_shuffled)))
    test_prot_ids_list = prot_ids_list_shuffled[:split_test_idx]

    split_train_idx = int(round((0.25 + 0.7 * 0.75) * len(prot_ids_list_shuffled)))
    train_prot_ids_list = prot_ids_list_shuffled[split_test_idx:split_train_idx]

    validate_prot_ids_list = prot_ids_list_shuffled[split_train_idx:]

    uniprot_id_index_test_val_train =  {
        "test": test_prot_ids_list, 
        "val": validate_prot_ids_list,
        "train": train_prot_ids_list
    }

    return uniprot_id_index_test_val_train
    
def extract_uniprot_ids_from_list(data_list):
    uniprot_ids = []
    for item in data_list:
        uniprot_ids.append(os.path.basename(item).split('-')[1])
    return uniprot_ids


def random_sample(input_list, x):
    if not 0 <= x <= 1:
        raise ValueError("Percentage must be between 0 and 1")

    sample_size = int(len(input_list) * x)
    return random.sample(input_list, sample_size)


def calculate_max_axis_histogram_from_uniprot_id_list(protein_atom_point_clouds_folder, prot_ids_list):
    lengths = []

    # Taking a random sample of 2% from the list
    sampled_prot_ids = random_sample(prot_ids_list, 0.1)

    # Adding a progress bar with tqdm
    for uniprot_id in tqdm(sampled_prot_ids, desc='Processing', unit='uniprot_id'):
        length = calculate_max_axis_from_uniprot(protein_atom_point_clouds_folder, uniprot_id)
        lengths.append(length)

    # Using plotly for the histogram
    plt.hist(lengths, bins=10)  # 'auto' lets Matplotlib decide the number of bins
    plt.show()


def get_dataset_index_simple_split(data_folder_name="protein_atom_point_clouds"):
    protein_atom_point_clouds_folder = os.path.join("data", data_folder_name)
    all_filenames_list = glob.glob(os.path.join(protein_atom_point_clouds_folder, "*.hdf5"))

    prot_ids_list = extract_uniprot_ids_from_list(all_filenames_list)
    
    uniprot_id_index_test_val_train = shuffle_and_split_uniprot_ids_simple(prot_ids_list)

    return uniprot_id_index_test_val_train

def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    
def uniprot_id_to_one_hot(full_uniprot_id_list, uniprot_id):
    # Initialize a one-hot encoded vector with all zeros
    one_hot_vector = [0] * len(full_uniprot_id_list)

    # Get the index of the uniprot_id in the list
    if uniprot_id in full_uniprot_id_list:
        index = full_uniprot_id_list.index(uniprot_id)
        # Set '1' at the corresponding position
        one_hot_vector[index] = 1

    return one_hot_vector

def yield_data_from_uniprot_id_list(uniprot_ids_list, full_uniprot_id_list, batch_size, grid_size=(100, 100, 100), num_channels=4):
    batch_X = []
    batch_Y = []
    for uniprot_id in uniprot_ids_list:
        X_4_channel_one_hot_3d_volume = get_4channel_cloud_data_from_uniprot(uniprot_id, grid_size=grid_size, num_channels=num_channels)
        Y_class_id_based_on_uniprot_to_one_hot = uniprot_id_to_one_hot(full_uniprot_id_list, uniprot_id)

        batch_X.append(X_4_channel_one_hot_3d_volume)
        batch_Y.append(Y_class_id_based_on_uniprot_to_one_hot)

        if len(batch_X) == batch_size:
            yield np.array(batch_X), np.array(batch_Y)
            batch_X, batch_Y = [], []

    # Yield any remaining data as the last batch
    if batch_X:
        yield np.array(batch_X), np.array(batch_Y)

def main():
    # Load the train val test split index of uniprot_ids
    data_split_index_filename = "data_split_uniprod_id_index.json"  # Added .json extension for clarity
    uniprot_id_index_test_val_train = {}

    if not os.path.exists(data_split_index_filename):
        uniprot_id_index_test_val_train = get_dataset_index_simple_split()
        save_json(uniprot_id_index_test_val_train, data_split_index_filename)
    else:  
        uniprot_id_index_test_val_train = load_json(data_split_index_filename)

    full_uniprot_ids_list_index_concatenated = uniprot_id_index_test_val_train["test"] + uniprot_id_index_test_val_train["val"] + uniprot_id_index_test_val_train["train"]
    print(len(full_uniprot_ids_list_index_concatenated))
    full_uniprot_ids_list_index_concatenated = list(set(full_uniprot_ids_list_index_concatenated))
    print(len(full_uniprot_ids_list_index_concatenated))

    # Build the model
    num_classes = len(set(full_uniprot_ids_list_index_concatenated))
    model = build_model(input_shape=(4, 100, 100, 100), num_classes=num_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Batch size
    batch_size = 32

    # Prepare the generators with batch size
    train_generator = yield_data_from_uniprot_id_list(uniprot_id_index_test_val_train["train"], full_uniprot_ids_list_index_concatenated, batch_size)
    val_generator = yield_data_from_uniprot_id_list(uniprot_id_index_test_val_train["val"], full_uniprot_ids_list_index_concatenated, batch_size)
    test_generator = yield_data_from_uniprot_id_list(uniprot_id_index_test_val_train["test"], full_uniprot_ids_list_index_concatenated, batch_size)

    # Determine steps per epoch and validation steps
    steps_per_epoch = len(uniprot_id_index_test_val_train["train"]) // batch_size
    validation_steps = len(uniprot_id_index_test_val_train["val"]) // batch_size

    # Train the model
    history = train_model(model, train_generator, val_generator, epochs=10, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    # Test the model
    evaluate_model(model, test_generator)


if __name__ == "__main__":
    main()
