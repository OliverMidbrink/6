import pandas as pd
import os
import json
import numpy as np

def main():
    data_path = os.path.join("data", "multisels_output")

    data = []
    iPPI_evaluations = []
    smiless = []

    for file_name in sorted(os.listdir(data_path)):
        full_file_name = os.path.join(data_path, file_name)
        if "eval_PPIs.json" in file_name:
            continue
        
        if "iPPI_evaluations" in file_name:
            continue

        try:
            with open(full_file_name, "r") as file:
                iPPI_probs = json.load(file)["iPPI_probs"]

            data.append(iPPI_probs)
            smiless.append(file_name.replace(".json", ""))
        except:
            print(file_name)
            os.remove(full_file_name)

    
    with open(os.path.join(data_path, "iPPI_evaluations.json"), "r") as file:
        iPPI_evaluations = json.load(file)["iPPI_evaluations"]
    
    iPPI_evaluations = np.array(iPPI_evaluations)
    data = np.array(data).transpose()

    mol_scores = np.matmul(iPPI_evaluations, data)

    best_indexes = np.argsort(mol_scores)[::-1][:10]

    data = {"IDs": range(len(smiless)), "Molecules": smiless, "Molecule Scores": mol_scores}
    df = pd.DataFrame(data=data)
    df.to_csv("multisels/MoleculeEvaluationResultsOSK.csv")
    
    print(f"Best molecules had scores: {mol_scores[best_indexes]}")

    for idx in best_indexes:
        print("Molecule Smiles:")
        print(smiless[idx])

        print("Molecule Score: ")
        print(mol_scores[idx])
    


if __name__ == "__main__":
    main()