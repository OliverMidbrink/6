import requests
from time import sleep
import pandas as pd
import openai, sys, time, random, re
import numpy as np
import json
from openai import OpenAI
from data_generator import *
import os

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)


def ask_gpt(input_text, prompt, model):
    gpt_response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ],
        model=model,
        temperature=0.0
        
    )
    return gpt_response.choices[0].message.content

def fetch_uniprot_data(protein_id):
    """
    Fetches data from UniProt for the given protein ID.
    
    Args:
    protein_id (str): UniProt ID of the protein.
    
    Returns:
    dict: JSON response from UniProt containing the protein data.
    """
    try:
        url = f"https://www.ebi.ac.uk/proteins/api/proteins/{protein_id}"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()

            return data
        else:
            print(f"Failed to fetch data for: {protein_id}, Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred for UniProt ID {protein_id}: {e}")
        return None

def extract_info_from_json(json_data):
    data_str = ""
    data_str += json.dumps(json_data['protein'])
    data_str += json.dumps(json_data['gene'])
    return data_str

def main():
    all_iPPI_helpfullness_data = {}
    all_iPPI_helpfullness_data_filename = "all_iPPI_ChatGPT_4_analysis_data.json"

    if os.path.exists(all_iPPI_helpfullness_data_filename):
        with open(all_iPPI_helpfullness_data_filename, "r") as file:
            all_iPPI_helpfullness_data = json.load(file)

    synthetic_iPPI_data = load_synthetic_iPPI_data()

    for iPPI_item in synthetic_iPPI_data.values():
        if iPPI_item["compound_id"] in all_iPPI_helpfullness_data.keys():
            # Skip if already processed before
            print("Already processed {}".format(iPPI_item["compound_id"]))
            continue

        uniprot_id_A, uniprot_id_B = iPPI_item["proteins"]
        SMILES = iPPI_item["SMILES"][0]
        random_iPPI_key = iPPI_item["compound_id"]
        print("Checking interaction {}".format(random_iPPI_key))

        try: 
            data_A = fetch_uniprot_data(uniprot_id_A)
            data_str_A = extract_info_from_json(data_A)

            data_B = fetch_uniprot_data(uniprot_id_B)
            data_str_B = extract_info_from_json(data_B)

        except Exception as e:
            continue

        total_smiles_available_drug_score = 0
        total_smiles_iterated = 0
        input_text = SMILES + ": Is this SMILES compound an active ingredient in a sold medical compound? Give a short 0 to 100 where 100 is sold. Nothing else."
        print(input_text)

        while total_smiles_iterated < 2:
            try:
                output_drug_availability_score = float(ask_gpt(input_text, "", "gpt-3.5-turbo-1106")) / 100
                total_smiles_iterated += 1
                total_smiles_available_drug_score += output_drug_availability_score
                print(output_drug_availability_score)
            except Exception as e:
                print(e)
        
        avg_SMILES_availability = total_smiles_available_drug_score / total_smiles_iterated
        print("Avg SMILES availability: {}, total_smiles_iterated: {}".format(avg_SMILES_availability, total_smiles_iterated))



        input_text = "Would these two proteins [{}, {}] interact and how helpful would inhibiting this interaction be for upregulating Oct4, Sox2 and Klf4?. You have to give a number from 0 to 100. Only respond with this number.".format(data_str_A, data_str_B)
        print(input_text)

        total_iPPI_helpfullness = 0
        iterated_proteins = 0
        while iterated_proteins < 2:
            try:
                output_iPPI_helpfullness = float(ask_gpt(input_text, "", "gpt-3.5-turbo-1106")) / 100
                iterated_proteins += 1
                total_iPPI_helpfullness += output_iPPI_helpfullness
                print(output_iPPI_helpfullness)
            except Exception as e:
                print(e)
        
        avg_iPPI_helpfullness = total_iPPI_helpfullness / iterated_proteins
        print("Avg iPPI helpfullness: {}, iterated_proteins: {}".format(avg_iPPI_helpfullness, iterated_proteins))
            
        all_iPPI_helpfullness_data[random_iPPI_key] = {
            "uniprot_id_A": uniprot_id_A,
            "uniprot_id_B": uniprot_id_B,
            "SMILES": SMILES,
            "AVG_SMILES_AVAILABILITY_AS_DRUG": avg_SMILES_availability,
            "AVG_iPPI_helpfullness_for_OSK_upregulation": avg_iPPI_helpfullness,
        }

        with open(all_iPPI_helpfullness_data_filename, "w") as file:
            json.dump(all_iPPI_helpfullness_data, file)

if __name__ == "__main__":
    main()