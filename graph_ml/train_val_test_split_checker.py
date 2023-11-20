import json


def get_uniprots_from_protein_pairs_and_labels(protein_pairs_and_labels):
    uniprots = []

    for uniprot_1, uniprot_2, label in protein_pairs_and_labels:
        uniprots.append(uniprot_1)
        uniprots.append(uniprot_2)
    
    return set(uniprots)

def load_protein_pairs_and_labels_from_file(file_name):
    protein_pairs_and_labels = []

    with open(file_name, "r") as file:
        protein_pairs_and_labels = json.load(file)["protein_pairs_and_labels"]

    return protein_pairs_and_labels


def check_for_overlap(train_set, val_set, test_set):
    # Find intersection between sets
    train_val_overlap = train_set.intersection(val_set)
    val_test_overlap = val_set.intersection(test_set)
    train_test_overlap = train_set.intersection(test_set)

    # Check for any overlap
    any_overlap = train_val_overlap or val_test_overlap or train_test_overlap

    # Return a dictionary with the overlaps
    return {
        'train_val_overlap': train_val_overlap,
        'val_test_overlap': val_test_overlap,
        'train_test_overlap': train_test_overlap,
        'any_overlap': any_overlap
    }

def main():
    train_file_name = "graph_ml/train_uniprot_pairs.json"
    uniprots_train = get_uniprots_from_protein_pairs_and_labels(load_protein_pairs_and_labels_from_file(train_file_name))
    print("Train_n_unique_uniprots: {}.".format(len(uniprots_train)))

    val_file_name = "graph_ml/val_uniprot_pairs.json"
    uniprots_val = get_uniprots_from_protein_pairs_and_labels(load_protein_pairs_and_labels_from_file(val_file_name))
    print("Val_n_unique_uniprots: {}".format(len(uniprots_val)))

    test_file_name = "graph_ml/test_uniprot_pairs.json"
    uniprots_test = get_uniprots_from_protein_pairs_and_labels(load_protein_pairs_and_labels_from_file(test_file_name))
    print("Test_n_unique_uniprots: {}".format(len(uniprots_test)))

    overlap_results = check_for_overlap(uniprots_train, uniprots_val, uniprots_test)
    if overlap_results['any_overlap']:
        print("There is overlap between the sets.")
        print(overlap_results)
    else:
        print("No overlap between the sets.")


if __name__ == "__main__":
    main()