import os
import numpy as np
import h5py
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from tqdm import tqdm
import json
from rdkit import Chem
from rdkit.Chem import AllChem


def csr_graph_from_point_cloud(atom_point_cloud, STANDARD_BOND_LENGTH=1.5):
    # Build a k-d tree for quick nearest-neighbor lookup
    tree = cKDTree(atom_point_cloud[:,1:])
    
    # Query the tree for pairs within the bond length
    pairs = tree.query_pairs(r=STANDARD_BOND_LENGTH)
    
    # Create row index and column index arrays for CSR format
    row_idx = np.array([pair[0] for pair in pairs])
    col_idx = np.array([pair[1] for pair in pairs])
    
    # Create data array for CSR format (all ones, assuming single bond)
    data = np.ones(len(pairs))
    
    # Create the CSR matrix
    csr_graph = csr_matrix((data, (row_idx, col_idx)), shape=(len(atom_point_cloud), len(atom_point_cloud)))
    
    return csr_graph


def smiles_to_atom_cloud(smile):
    """
    This function takes a SMILES string and generates a 3D atom cloud as a NumPy array,
    which includes the converted atom type as float and 3D coordinates for each atom in the molecule.
    Hydrogens are used for generating the conformation but are not included in the
    final NumPy array unless specified.
    
    Args:
    smile (str): A SMILES string representing a molecule.
    include_hydrogens (bool): If True, include hydrogen atoms in the final output.
    
    Returns:
    np.array: A NumPy array with each row representing an atom type as float and the (x, y, z) coordinates.
    """
    # Atom type to float mapping
    atom_type_to_float = {
        'C': 1.0,  # Carbon
        'N': 2.0,  # Nitrogen
        'O': 3.0,  # Oxygen
        'S': 4.0,  # Sulfur
        'P': 5.0,  # Phosphorus
        'F': 6.0,  # Fluorine
        'Cl': 7.0, # Chlorine
        'Br': 8.0, # Bromine
        'I': 9.0,  # Iodine
    }
    
    # Convert the SMILES string to a molecule object
    molecule = Chem.MolFromSmiles(smile)
    if not molecule:
        raise ValueError("Invalid SMILES string")
    
    # Add hydrogens to the molecule
    molecule = Chem.AddHs(molecule)
    
    # Generate a 3D conformation for the molecule
    AllChem.EmbedMolecule(molecule, AllChem.ETKDG())
    
    # Minimize the energy of the conformation
    AllChem.UFFOptimizeMolecule(molecule)
    
    # Extract the atom types and 3D coordinates of the atoms
    conf = molecule.GetConformer()
    atom_cloud_data = []
    for idx, atom in enumerate(molecule.GetAtoms()):
        if atom.GetSymbol() != 'H':
            atom_type = atom_type_to_float[atom.GetSymbol()]
            position = conf.GetAtomPosition(idx)
            atom_cloud_data.append((atom_type, position.x, position.y, position.z))

    # Convert the atom cloud data to a NumPy array
    atom_cloud_array = np.array(atom_cloud_data, dtype=float)
    
    return atom_cloud_array

def save_to_hdf5(csr_graph, features, file_path):
    with h5py.File(file_path, 'w') as f:
        # Save the features matrix
        f.create_dataset('features', data=features)
        # Save the CSR components
        f.create_dataset('data', data=csr_graph.data)
        f.create_dataset('indices', data=csr_graph.indices)
        f.create_dataset('indptr', data=csr_graph.indptr)
        f.create_dataset('shape', data=csr_graph.shape)


def main():
    output_dir_path = "data/mol_graphs"
    molecule_smiles = "DLiP_rule_of_5_compound_data.json"
    DLiP_data = {}

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    with open(molecule_smiles, "r") as file:
        DLiP_data = json.load(file)
    
    smiles = [x["SMILES"][0] for x in DLiP_data.values()]

    for smile in tqdm(smiles, desc="Loading smiles", unit="molecules"):
        atom_cloud = smiles_to_atom_cloud(smile)
        csr_graph = csr_graph_from_point_cloud(atom_cloud)

        atom_point_cloud_atom_types = atom_cloud[:, 0] # get the atom types
        n_atom_types = 9

        # One-hot encode the atom types
        features = np.eye(n_atom_types)[atom_point_cloud_atom_types.astype(int) - 1]        
        output_file_path = os.path.join(output_dir_path, '{}_graph.hdf5'.format(smile))
        save_to_hdf5(csr_graph, features, output_file_path)

if __name__ == "__main__":
    main()