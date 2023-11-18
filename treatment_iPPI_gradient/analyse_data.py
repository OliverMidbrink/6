import json
import pandas as pd

def load_data(filename = "all_iPPI_ChatGPT_4_analysis_data.json"):
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def load_synthetic_data(filename = "../DLiP_rule_of_5_compound_data.json"):
    with open(filename, "r") as file:
        data = json.load(file)
    return data

def main():
    data_gpt = load_data()
    data_full_DLiP = load_synthetic_data()

    rows = []

    for x in data_gpt:
        # Create data rows

        data_item = data_gpt[x]

        PPI_id = x
        uniprot_id_A = data_item["uniprot_id_A"]
        uniprot_id_B = data_item["uniprot_id_B"]
        SMILES = data_item["SMILES"]
        AVG_SMILES_AVAILABILITY_AS_DRUG = data_item["AVG_SMILES_AVAILABILITY_AS_DRUG"]
        AVG_iPPI_helpfullness_for_OSK_upregulation = data_item["AVG_iPPI_helpfullness_for_OSK_upregulation"]
        COMBINED_SCORE = AVG_SMILES_AVAILABILITY_AS_DRUG * AVG_iPPI_helpfullness_for_OSK_upregulation

        rows.append([PPI_id, uniprot_id_A, uniprot_id_B, SMILES, AVG_SMILES_AVAILABILITY_AS_DRUG, AVG_iPPI_helpfullness_for_OSK_upregulation, COMBINED_SCORE])

    df = pd.DataFrame(rows, columns=["PPI_id_DLiP", "uniprot_id_A", "uniprot_id_B", "SMILES", "AVG_SMILES_AVAILABILITY_AS_DRUG", "AVG_iPPI_helpfullness_for_OSK_upregulation", "COMBINED_SCORE"])
    
    sorted_by_score = df.sort_values(by="COMBINED_SCORE", ascending=False)

    sorted_by_score.to_csv('chat_gpt_analysis.csv', index=False)


    print(sorted_by_score.head(50))



if __name__ == "__main__":
    main()