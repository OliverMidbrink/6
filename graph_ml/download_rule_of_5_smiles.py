import requests
from time import sleep
from tqdm import tqdm

def fetch_compounds(listkey, start, count):
    compounds_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/{listkey}/cids/JSON?Start={start}&Count={count}"
    return requests.get(compounds_url).json()

def fetch_properties(cid):
    properties_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularWeight,HBondDonorCount,HBondAcceptorCount,XLogP/JSON"
    return requests.get(properties_url).json()

# Placeholder search query
search_query = 'C'

# Adjust the batch size as needed
batch_size = 100

# Initialize tqdm progress bar
progress_bar = tqdm(desc='Downloading SMILES')

# The initial request to start the search
initial_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/substructure/smiles/{search_query}/JSON?Add3D=true"
initial_response = requests.get(initial_url)
if initial_response.status_code == 200 and 'Waiting' in initial_response.json():
    listkey = initial_response.json()['Waiting']['ListKey']
    # Poll using the listkey until the results are ready
    ready = False
    while not ready:
        sleep(5)  # Wait for 5 seconds before checking the status
        status_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/{listkey}/cids/JSON"
        status_response = requests.get(status_url)
        if status_response.status_code == 200:
            ready = True
            # Fetch compounds in batches
            total_compounds = status_response.json()['IdentifierList']['CID']
        total_count = len(total_compounds)

        # Loop over all compounds in batches
        for start_index in tqdm(range(0, total_count, batch_size), desc='Downloading SMILES'):
            end_index = min(start_index + batch_size, total_count)  # Ensure we don't go beyond the total count
            compounds_batch = fetch_compounds(listkey, start_index, end_index - start_index)
            cids = compounds_batch['IdentifierList']['CID']
            
            # Fetch properties and write SMILES for each CID
            for cid in cids:
                properties_response = fetch_properties(cid)
                properties = properties_response['PropertyTable']['Properties'][0]
                if check_lipinski(properties):
                    with open('lipinski_rule_of_5_smiles.txt', 'a') as file:
                        file.write(properties['CanonicalSMILES'] + '\n')

            progress_bar.update(end_index - start_index)
progress_bar.close()
print("SMILES saved to lipinski_rule_of_5_smiles.txt")
