import json
import os
import random

def main():
    output_dir_path = "data/mol_graphs"
    molecule_smiles = "DLiP_rule_of_5_compound_data.json"
    DLiP_data = {}

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    with open(molecule_smiles, "r") as file:
        DLiP_data = json.load(file)
    
    uniprot_ids = list(set([prot for x in DLiP_data.values() for prot in x["proteins"]]))
    random.shuffle(uniprot_ids)

    # 1 divide the DLiP_data
    keys = DLiP_data.keys()
    random.shuffle(keys)


    train_val_split = int(len(keys) * 0.8)
    val_test_split = int(len(keys) * 0.9)
    train_keys = keys[:train_val_split]
    val_keys = keys[train_val_split:val_test_split]
    test_keys = keys[val_test_split:]



    # 2 divide all the overlaps

if __name__ == "__main__":
    main()