import requests
from time import sleep
import pandas as pd
import openai, sys, time, random, re
import numpy as np
import json
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="",
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
    return ask_gpt(data, "Given the following input data and your knowledge of biology surrounding this input data, how likely is this to relate to up or downregulation of Oct4, Sox2 and Klf4 as in the yamanaka transcription factors. Give a 0-100 estimate of interaction relevance. Also, would it be useful to screen? Summurize this in as a python array with interaction relevance and usefulness (0-100 percent scale) as the elements of the vector. But only have the list and no variable names. also no comments. Just the brackets with the values inside. Please thank you. Please also add how much it would upregulate or downregulate the OSK factors as a percent number in the effect. Tha is up or down 10% for example and negative reuglation would be -10. thanks", "gpt-4-1106-preview")

def extract_values_from_gpt_output(text):
    pattern = r'```python\n(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None  # or an empty string if you prefer

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

analysis_output_json = {}

# Now iterate over each UniProt ID and fetch the data
for protein_id in uniprot_ids[:10]:
    try:
        data = fetch_uniprot_data(protein_id)
        data_str = extract_info_from_json(data)
        print(data_str)

        print("sdf")
        ""
        analysis_text = analyze_protein_gpt_text_output(data_str)
        print(analysis_text)
        values = extract_values_from_gpt_output(analysis_text)
        
        analysis_output_json[protein_id] = analysis_text

        print(values)

        with open("gpt_based_analysis_output", "w") as file:
            json.dump(analysis_output_json, file)

        sleep(1)  # Sleep to respect the API's rate limit
    except Exception as e:
        print(e)

with open("data.json", "r") as file:
    json = json.load(file)
    print(json['protein'])
    print(json['gene'])