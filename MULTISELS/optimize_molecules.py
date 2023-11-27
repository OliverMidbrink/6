import json
import os
from iPPI_predict import get_model, get_combined_graph, predict
from tqdm import tqdm

def get_targets_iPPIs(file_path):
    with open(file_path, "r") as file:
        return json.load(file)["tuples"]

def get_smiles():
    smiles = set()
    for file_name in os.listdir("data/mol_graphs"):
        smile = file_name.split("_graph")[0]
        smiles.add(smile)
    return sorted(list(smiles))


def main():
    PPI_valuations = get_targets_iPPIs("MULTISELS/OSK_upreg_2_neighbors_chatGPT3_turpo_1106.json")

    data_path = os.path.join("data", "multisel_output")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    with open(os.path.join(data_path, "eval_PPIs.json"), "w") as file:
        json.dump({"eval_PPIs": PPI_valuations}, file)

    model = get_model()
    smiles = get_smiles()

    file_names = os.listdir(data_path)
    files_smiles = [file_name.replace(".json", "") for file_name in file_names]

    for smile in tqdm(smiles, desc="Getting smile scores", unit="smiles"):
        if not smile in files_smiles:
            graphs = []
            for PPI in PPI_valuations:
                graphs.append(get_combined_graph(PPI[0], PPI[1], smile))

            with open(os.path.join(data_path, smile + ".json"), "w") as file:
                json.dump({"iPPI_probs": list([float(x[0]) for x in predict(graphs, model)])}, file)
    

if __name__ == "__main__":
    main()