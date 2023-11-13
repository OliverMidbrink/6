from download_dlip_rule_of_5_compound_ids import format_seconds
import requests, json, os, time, math
from bs4 import BeautifulSoup

def get_uniprot_id(protein_name):
    query_url = "https://rest.uniprot.org/uniprotkb/search?fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength&query=%28{}%29organism_name=%28Human%29".format(protein_name.replace(" ","+"))

    response = requests.get(query_url)
    json_data = json.loads(response.content)
    uniprot_id = json_data["results"][0]["primaryAccession"]
    return uniprot_id



def get_compound_data(compound_id):
    try:
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
                    protein_name = tr.find_all("td")[1].text
                    uniprot_id = get_uniprot_id(protein_name)
                    compound_info_dict["proteins"].append(uniprot_id)

            # Find and filter <tr> elements based on some criteria
            tr_elements = soup.find_all('tr')  # Replace 'your-class-name' with the actual class name

            # Print the filtered <tr> elements
            for tr in tr_elements:
                if "Canonical SMILES(RDKit)" in tr.text:
                    can_smiles = tr.find_all("td")[1].text
                    compound_info_dict["SMILES"].append(can_smiles)

                if "SMILES(SDF)" in tr.text:
                    sdf_smiles = tr.find_all("td")[1].text
                    compound_info_dict["SMILES"].append(sdf_smiles)
        else:
            print(f"Request failed with status code: {response.status_code}")

        return compound_info_dict
    except:
        print("Could not get human interaction data for compound {}".format(compound_id))

def load_json(filename):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except:
        return None


def save_json(json_data, filename):
    try:
        with open(filename, "w") as file:
            return json.dump(json_data, file)
    except:
        return None

def convert_filesize_to_human_readable(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def download_all_rule_of_five_compounds_data(output_filename = "DLiP_rule_of_5_compound_data.json", rule_of_5_compound_ids_filename="DLiP_PPI_DB_rule_of_5_compound_ids.json"):
    # Get compound ids     
    rule_of_5_compound_ids = []

    if os.path.exists(rule_of_5_compound_ids_filename):
        with open(rule_of_5_compound_ids_filename, "r") as file:
            json_data = json.load(file)
            rule_of_5_compound_ids = json_data["rule_of_5_compound_ids"]
    else:
        print("You have to run download_dlip_rule_of_5_compound_ids.py")
        return None


    # Get compound data

    # Load if data exists
    if os.path.exists(output_filename):
        compound_data_dict = load_json(output_filename)
    else:
        compound_data_dict = {}
        save_json(compound_data_dict, output_filename)

    last_completion_time = time.time()
    last_file_size = os.path.getsize(output_filename)

    compounds_iterated = 0
    for compound_id in rule_of_5_compound_ids - compound:
        compounds_iterated += 1
        if not compound_id in compound_data_dict:
            compound_data = get_compound_data(compound_id)
            if compound_data:
                eta = (time.time() - last_completion_time) * (len(rule_of_5_compound_ids) - compounds_iterated)
                last_completion_time = time.time()
                eta_str = format_seconds(eta)
                projecte_filesize = (os.path.getsize(output_filename) - last_file_size) * (len(rule_of_5_compound_ids) - compounds_iterated)
                last_file_size = os.path.getsize(output_filename)
                projecte_filesize_str = convert_filesize_to_human_readable(projecte_filesize)
                print("Downloading and saving data for compound of id {} to file {}. ETA is {}. Projected file size is {}".format(compound_id, output_filename, eta_str, projecte_filesize_str))
                compound_data_dict[compound_id] = compound_data

                save_json(compound_data_dict, output_filename)
        else:
            print("Compound data for compound of id {} already saved to {}".format(compound_id, output_filename))

    return True



download_all_rule_of_five_compounds_data()