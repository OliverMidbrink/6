import os
import json
import zipfile

def list_files_in_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        return zip_ref.namelist()

def explore_zip(path, file_structure):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                inner_path = os.path.join(path, file)
                if zipfile.is_zipfile(inner_path):  # Check if it is a zip within a zip
                    explore_zip(inner_path, file_structure)
                else:
                    file_structure.append(inner_path)

def create_file_index(base_path, index_filename):
    file_structure = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if zipfile.is_zipfile(item_path):
            explore_zip(item_path, file_structure)

    with open(index_filename, 'w') as f:
        json.dump(file_structure, f)


if __name__ == "__main__":
    index_filename = 'pubchem_assays_file_index.json'
    if not os.path.exists(index_filename):
        pubchem_assays_path = 'pubchem_assays'
        create_file_index(pubchem_assays_path)
    else:
        print("Already exists a {} file".format(index_filename))
