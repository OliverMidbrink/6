import requests, sys, json, time
import pandas as pd
from bs4 import BeautifulSoup

def get_smiles_value(url):
    # Send a GET request to the webpage
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find all table elements on the page
        smiles = soup.find('pre').text
        return smiles
    else:
        raise Exception(f"Webpage request was unsuccessful. Status code: {response.status_code}")

def get_ppi_values(url):
    # Send a GET request to the webpage
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find all table elements on the page
        tables = soup.find_all("div")

        for table in tables:
            try:
                if table.find("h5").text == "Pharmacological data":
                    # Find the data
                    tbody = table.find("tbody")

                    values = []

                    for tr in tbody.find_all("tr"):
                        tds = tr.find_all("td")
                        target, activity_type, activity = tds[2].text, tds[7].text, tds[8].text
                        target_prot_gene_name = target.split(" ")[0]
                        target_protid = target.split(" ")[1].strip()
                        value = {
                            "target_prot_gene_name": target_prot_gene_name,
                            "target_protid": target_protid,
                            "activity_type": activity_type,
                            "activity": activity
                        }
                        values.append(value)
                    
                    return values
            except:
                pass
    else:
        raise Exception(f"Webpage request was unsuccessful. Status code: {response.status_code}")

def get_compound_info(compound_number):
    compound_data = {}
    compound_data["SMILES"] = get_smiles_value("https://ippidb.pasteur.fr/compounds/{}#compound".format(compound_number))
    compound_data["PPI_VALUES"] = get_ppi_values("https://ippidb.pasteur.fr/compounds/{}#pharmacology".format(compound_number))

    return compound_data


def get_number_of_compounds():
    response = requests.get("https://ippidb.pasteur.fr/compounds/")
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        # Find all table elements on the page
        spans = soup.find_all("span")

        for span in spans:
            if "compounds found" in span.text.lower():
                return int(span.text.split()[0])
    
    return None

def save_json(json_data, filename):
    with open(filename, "w") as file:
        json.dump(json_data, file)


def load_json(filename):
    with open(filename, "r") as file:
        json_data = json.load(file)
        return json_data

def is_already_read(filename, key_number):
    json_data = load_json(filename)
    if str(key_number) in json_data:
        return True
    else:
        return False

def compound_data_from_disk(filename, key_number):
    json_data = load_json(filename)
    if str(key_number) in json_data:
        return json_data[str(key_number)]
    else:
        return None


def format_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    if not parts:
        return "0 seconds"

    return ", ".join(parts)

def get_all_compound_data(filename):
    full_json_compound_data = {}

    # Get number of compounds
    n_compounds = get_number_of_compounds()

    loading_time = -1
    for x in range(1, n_compounds + 1):
        try:
            if not is_already_read(filename, x):
                start_time = time.time()

                eta = format_seconds(loading_time * (n_compounds - x))
                print("Getting compound data for {} out of {}. {}".format(x, n_compounds, "ETA is {}".format(eta) if loading_time != -1 else ""))
                compound_data = get_compound_info(x)

                full_json_compound_data[x] = compound_data

                save_json(full_json_compound_data, filename)

                loading_time = time.time() - start_time
            else:
                full_json_compound_data[x] = compound_data_from_disk(filename, x)
                print("Already loaded data for molecule of id {}".format(x))
        except Exception as e:
            print(e)

get_all_compound_data("iPPI-DB.json")