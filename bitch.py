import requests
from time import sleep
import pandas as pd

# Load the Excel file
file_path = '13321_2023_720_MOESM1_ESM.xlsx'
df = pd.read_excel(file_path)
uniprot_ids = df['Uniprot'].tolist()

def fetch_uniprot_data(protein_id):
    try:
        print(f"Fetching data for UniProt ID: {protein_id}")
        url = f"https://www.ebi.ac.uk/proteins/api/proteins/{protein_id}"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print(f"Data successfully fetched for: {protein_id}")
            return response.json()
        else:
            print(f"Failed to fetch data for: {protein_id}, Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred for UniProt ID {protein_id}: {e}")
        return None

def check_for_signalling(protein_data, term):
    print(f"Checking for signalling related to: {term}")
    if term.lower() in str(protein_data).lower():
        print(f"Signalling related to {term} found")
        return True
    print(f"No signalling related to {term} detected")
    return False

# Replace this with your actual list of UniProt IDs
unique_proteins = uniprot_ids  # Replace with your list of UniProt IDs

signalling_related_proteins = {
    'Oct4': [],
    'Sox2': [],
    'Klf4': []
}

# This is for testing purposes, to avoid making too many requests
# Remove slicing for the full run
batch = unique_proteins[:10]

for protein_id in batch:
    protein_data = fetch_uniprot_data(protein_id)
    sleep(1)  # Sleep to respect the API's rate limit
    if protein_data:
        for term in signalling_related_proteins.keys():
            if check_for_signalling(protein_data, term):
                signalling_related_proteins[term].append(protein_id)

print("Analysis complete. Results:")
print(signalling_related_proteins)
