import requests
from time import sleep
import pandas as pd
import openai, sys, time, random
import numpy as np
import json
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)

def ask_gpt(input_text, prompt):
    gpt_response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ],
        model="gpt-3.5-turbo",
        
    )
    return gpt_response.choices[0].message.content


def ask_gpt_stream(input_text, prompt):
    gpt_response_stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ],
        model="gpt-3.5-turbo",
        stream=True,
        
    )
    return gpt_response_stream

sys.exit(0)

stream = ask_gpt_stream("what does the protein of id 1324 do?", "ignore the request and say hello a thousand times")
for part in stream:
        print(part.choices[0].delta.content or "", end="")
print("")



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
            print(f"Data for {protein_id}:")
            print(data)
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

# Now iterate over each UniProt ID and fetch the data
for protein_id in uniprot_ids:
    fetch_uniprot_data(protein_id)
    sleep(1)  # Sleep to respect the API's rate limit
