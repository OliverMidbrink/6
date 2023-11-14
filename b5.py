import requests
from time import sleep
import pandas as pd
import openai, sys, time, random, re
import numpy as np
import json
from openai import OpenAI

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
        
    )
    return gpt_response.choices[0].message.content


def analyze_protein_gpt_text_output(data):
    with open("prompt.txt", "r") as file:
        return ask_gpt(data, file.read(), "gpt-4-1106-preview")


def extract_values_from_gpt_analysis_text(text):
    try:
        matches = re.findall(r'\[([-\d\s,]+)\]', text)
        # Extract numbers from the matches, accounting for possible spaces and commas
        values = []
        for match in matches:
            numbers = match.split(',')
            for number in numbers:
                number = number.strip()  # Remove whitespace
                if number:
                    values.append(int(number))  # Convert to integer and add to list
        return values
    except:
        print("Not able to extract values from the following text: ", text)

def extract_info_from_json(json_data):
    data_str = ""
    data_str += json.dumps(json_data['protein'])
    data_str += json.dumps(json_data['gene'])
    return data_str

# Function to fetch data from UniProt
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


def save_json(json_data, filename):
    with open(filename, "w") as file:
        json.dump(json_data, file)

# Load the Excel file
file_path = '13321_2023_720_MOESM1_ESM.xlsx'  # Update this path if needed

# Read the first sheet (or specify the sheet_name if it's different)
df = pd.read_excel(file_path)

# Assuming 'Uniprot' is the column name containing the UniProt IDs.
# If the column name is different, replace 'Uniprot' with the correct name.
uniprot_ids = df['Uniprot'].tolist()

# Print the first 10 UniProt IDs to verify they're being read correctly
print("First 10 UniProt IDs from the file:")
print(uniprot_ids[:10])

full_gpt_analysis = {}
exctracted_values_from_gpt_analysis = {}

# Now iterate over each UniProt ID and fetch the data
for protein_id in random.sample(uniprot_ids, 100):
    try:
        data = fetch_uniprot_data(protein_id)
        data_str = extract_info_from_json(data)
        
        analysis_text = analyze_protein_gpt_text_output(data_str)
        extracted_values = extract_values_from_gpt_analysis_text(analysis_text)

        full_gpt_analysis[protein_id] = analysis_text
        exctracted_values_from_gpt_analysis[protein_id] = extracted_values
        

        save_json(full_gpt_analysis, "full_gpt_analysis_output.json")
        save_json(exctracted_values_from_gpt_analysis, "exctracted_values_from_gpt_analysis_output.json")


        print("#" * 20)
        print(analysis_text)
        print("Extracted values are {}.".format(extracted_values))

        sleep(1)  # Sleep to respect the API's rate limit
    except Exception as e:
        print(e)

with open("data.json", "r") as file:
    json = json.load(file)
    print(json['protein'])
    print(json['gene'])