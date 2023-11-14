import os
import glob
import gzip
import shutil
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
from tqdm import tqdm

def extract_files_by_uniprot_id_and_return_pdb_filenames(folder_path, uniprot_id, output_folder=None):
    if output_folder is None:
        output_folder = folder_path

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pattern = os.path.join(folder_path, f"AF-{uniprot_id}-F*-model_v4.pdb.gz")
    files = glob.glob(pattern)
    pdb_file_paths = []

    for file in files:
        if file.endswith('.gz'):
            output_file = os.path.join(output_folder, os.path.basename(file)[:-3])  # remove .gz from filename
            with gzip.open(file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extracted: {output_file}")
            if output_file.endswith('.pdb'):
                pdb_file_paths.append(output_file)

    return pdb_file_paths

def get_structure_from_pdb_path(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", pdb_path)
    return structure

def get_protein_atom_coordinates(pdb_path):
    structure = get_structure_from_pdb_path(pdb_path)
    protein_coordinates_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_type_numeric = get_atom_type_numeric(atom.element.strip())
                    atom_data = [atom_type_numeric, *atom.get_coord()]
                    protein_coordinates_list.append(atom_data)
    return protein_coordinates_list

atom_type_mapping = {
    'C': 1,  # Carbon
    'N': 2,  # Nitrogen
    'O': 3,  # Oxygen
    'S': 4,  # Sulfur
    'H': 5,  # Hydrogen
    # Add more elements if needed
}

def get_atom_type_numeric(atom_type):
    return atom_type_mapping.get(atom_type, 0)

def plot_protein_atoms(atom_coordinates):
    # Define colors for different atom types (you can expand this dictionary)
    atom_colors = {
        'C': 'black',  # Carbon - black
        'N': 'blue',   # Nitrogen - blue
        'O': 'red',    # Oxygen - red
        'S': 'yellow', # Sulfur - yellow
        'H': 'gray'    # Hydrogen - gray
        # Add more elements if needed
    }

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for atom in atom_coordinates:
        atom_type, x, y, z = atom
        color = atom_colors.get(atom_type, 'green')
        ax.scatter(x, y, z, c=color, marker='o')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()

def save_to_hdf5(data, file_path, dataset_name):
    with h5py.File(file_path, 'w') as h5file:
        h5file.create_dataset(dataset_name, data=data)

def extract_files_by_uniprot_id_and_return_pdb_filenames(folder_path, uniprot_id, output_folder=None):
    if output_folder is None:
        output_folder = folder_path

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pattern = os.path.join(folder_path, f"AF-{uniprot_id}-F*-model_v4.pdb.gz")
    files = glob.glob(pattern)
    pdb_file_paths = []

    for file in files:
        if file.endswith('.gz'):
            output_file = os.path.join(output_folder, os.path.basename(file)[:-3])  # remove .gz from filename
            with gzip.open(file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extracted: {output_file}")
            if output_file.endswith('.pdb'):
                pdb_file_paths.append(output_file)

    return pdb_file_paths

def main(folder_path):
    pdb_gz_files = glob.glob(os.path.join(folder_path, "*.pdb.gz"))
    uniprot_ids = {os.path.basename(f).split('-')[1] for f in pdb_gz_files}

    pbar = tqdm(uniprot_ids, desc="Processing proteins", unit="protein")
    for uniprot_id in pbar:
        try:
            pbar.set_postfix({"UniProt ID": uniprot_id})
            pdb_paths = extract_files_by_uniprot_id_and_return_pdb_filenames(folder_path, uniprot_id)
            for pdb_path in pdb_paths:
                protein_coordinates = get_protein_atom_coordinates(pdb_path)
                hdf5_file_path = os.path.join("data", "protein_atom_point_clouds", f"{os.path.basename(pdb_path).replace('.pdb', '')}_atom_cloud.hdf5")
                save_to_hdf5(protein_coordinates, hdf5_file_path, 'atom_coordinates')
                pbar.set_description(f"Saved {os.path.basename(pdb_path)}")
        except Exception as e:
            pbar.set_description(f"Error with {uniprot_id}")
            print(f"An error occurred while processing {uniprot_id}: {e}")

if __name__ == "__main__":
    data_folder_path = "data/UP000005640_9606_HUMAN_v4"
    main(data_folder_path)
