import numpy as np
import os
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import h5py


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

def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as h5file:
        print("Reading File")
        return np.array(h5file['atom_coordinates'])


def get_4_channel_grid_data_from_uniprot(uniprot_id, protein_atom_point_clouds_folder=os.path.join("..", "data", "protein_atom_point_clouds"), grid_size=(100, 100, 100), num_channels=4):
    # Only use the first fold of each uniprot
    protein_atom_point_cloud_filename = os.path.join(protein_atom_point_clouds_folder, "AF-{}-F1-model_v4_atom_cloud.hdf5".format(uniprot_id))
    
    coordinates = read_hdf5(protein_atom_point_cloud_filename)
    
    transformed_coordinates = apply_transformations(coordinates, grid_size)

    channels = np.zeros((num_channels, *grid_size))

    # Project coordinates into channels
    for coord in transformed_coordinates:
        atom_type = int(coord[0]) # Assuming the atom type is the first element
        if 1 <= atom_type <= num_channels:
            x, y, z = map(int, coord[1:])
            channels[atom_type - 1, x, y, z] = 1  # Mark the presence of the atom

    return channels

def smiles_to_atom_cloud(smiles):
    # Atom type to integer mapping (excluding hydrogen)
    atom_types = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'Cl': 6, 'Br': 7, 'I': 8}

    # Convert SMILES to a molecule object
    mol = Chem.MolFromSmiles(smiles)

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    # Extracting the 3D coordinates and encoded atom types, skipping hydrogens
    point_cloud = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue  # Skip hydrogen atoms
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        atom_symbol = atom.GetSymbol()
        atom_type = atom_types.get(atom_symbol, 8)  # Default to 8 if atom type not in the list
        point_cloud.append((atom_type, pos.x, pos.y, pos.z))

    return np.array(point_cloud)


def get_4_channel_grid_data_from_smiles(smiles, grid_size=(30, 30, 30), num_channels=9):
    # Only use the first fold of each uniprot
    
    coordinates = smiles_to_atom_cloud(smiles)
    
    transformed_coordinates = apply_transformations(coordinates, grid_size)

    channels = np.zeros((num_channels, *grid_size))

    # Project coordinates into channels
    for coord in transformed_coordinates:
        atom_type = int(coord[0]) # Assuming the atom type is the first element
        if 1 <= atom_type <= num_channels:
            x, y, z = map(int, coord[1:])
            channels[atom_type - 1, x, y, z] = 1  # Mark the presence of the atom

    return channels