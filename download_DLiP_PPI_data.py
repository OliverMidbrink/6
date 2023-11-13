import requests, json
from bs4 import BeautifulSoup

def get_uniprot_id(protein_name):
    query_url = "https://rest.uniprot.org/uniprotkb/search?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength&query=%28{}%29organism_name=%28Human%29".format(protein_name.replace(" ","+"))

    response = requests.get(query_url)
    json_data = json.loads(response.content)
    uniprit_id = json_data["results"][0]["primaryAccession"]
    return uniprit_id


def get_compound_info(compound_id):
    compound_info_dict = {}
    compound_info_dict["proteins"] = []
    compound_info_dict["SMILES"] = []
    compound_info_dict["compound_id"] = compound_id

    # Define the URL with the provided compound ID
    url = f"https://skb-insilico.com/dlip/compound/{compound_id}"

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find and filter <tr> elements based on some criteria
        tr_elements = soup.find_all('tr')  # Replace 'your-class-name' with the actual class name

        # Print the filtered <tr> elements
        for tr in tr_elements:
            if "Protein Name" in tr.text:
                print(tr.find_all("td")[0].text)
                print(tr.find_all("td")[1].text)  # Print the entire <tr> element including its children
                protein_name = tr.find_all("td")[1].text
                uniprot_id = get_uniprot_id(protein_name)
                compound_info_dict["proteins"].append(uniprot_id)

        # Find and filter <tr> elements based on some criteria
        tr_elements = soup.find_all('tr')  # Replace 'your-class-name' with the actual class name

        # Print the filtered <tr> elements
        for tr in tr_elements:
            if "Canonical SMILES(RDKit)" in tr.text:
                print("CAN SMILES")
                can_smiles = tr.find_all("td")[1].text
                print(can_smiles)  # Print the entire <tr> element including its children
                compound_info_dict["SMILES"].append(can_smiles)

            if "SMILES(SDF)" in tr.text:
                print("SMILES(SDF)")
                sdf_smiles = tr.find_all("td")[1].text
                print(sdf_smiles)  # Print the entire <tr> element including its children
                compound_info_dict["SMILES"].append(sdf_smiles)
    else:
        print(f"Request failed with status code: {response.status_code}")

    return compound_info_dict



def download_all_rule_of_five_compounds(output_filename, ppi_ids_filename):

    compound_id = "D00000"  # Replace with the compound ID you want to query
    compound_info = get_compound_info(compound_id)
    print(compound_info)