import requests, json, os, time

def get_data(start_id, length):
    # The URL from which we want to download data
    # Follow the rule of 5 (chemistry/pharmacology) to make it orally available
    url = "https://skb-insilico.com/dlip/compound/advanced-search?draw=2&columns%5B0%5D%5Bdata%5D=ppi_id&columns%5B0%5D%5Bname%5D=&columns%5B0%5D%5Bsearchable%5D=true&columns%5B0%5D%5Borderable%5D=false&columns%5B0%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B0%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B1%5D%5Bdata%5D=molecule_image&columns%5B1%5D%5Bname%5D=&columns%5B1%5D%5Bsearchable%5D=true&columns%5B1%5D%5Borderable%5D=false&columns%5B1%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B1%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B2%5D%5Bdata%5D=compoundproperty.rd_mw_freebase&columns%5B2%5D%5Bname%5D=&columns%5B2%5D%5Bsearchable%5D=true&columns%5B2%5D%5Borderable%5D=false&columns%5B2%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B2%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B3%5D%5Bdata%5D=compoundproperty.mw_monoisotopic&columns%5B3%5D%5Bname%5D=&columns%5B3%5D%5Bsearchable%5D=true&columns%5B3%5D%5Borderable%5D=false&columns%5B3%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B3%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B4%5D%5Bdata%5D=compoundproperty.cdk_xlogp&columns%5B4%5D%5Bname%5D=&columns%5B4%5D%5Bsearchable%5D=true&columns%5B4%5D%5Borderable%5D=false&columns%5B4%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B4%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B5%5D%5Bdata%5D=compoundproperty.mol_logp&columns%5B5%5D%5Bname%5D=&columns%5B5%5D%5Bsearchable%5D=true&columns%5B5%5D%5Borderable%5D=false&columns%5B5%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B5%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B6%5D%5Bdata%5D=compoundproperty.rd_hba&columns%5B6%5D%5Bname%5D=&columns%5B6%5D%5Bsearchable%5D=true&columns%5B6%5D%5Borderable%5D=false&columns%5B6%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B6%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B7%5D%5Bdata%5D=compoundproperty.hba_lipinski&columns%5B7%5D%5Bname%5D=&columns%5B7%5D%5Bsearchable%5D=true&columns%5B7%5D%5Borderable%5D=false&columns%5B7%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B7%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B8%5D%5Bdata%5D=compoundproperty.rd_hbd&columns%5B8%5D%5Bname%5D=&columns%5B8%5D%5Bsearchable%5D=true&columns%5B8%5D%5Borderable%5D=false&columns%5B8%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B8%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B9%5D%5Bdata%5D=compoundproperty.hbd_lipinski&columns%5B9%5D%5Bname%5D=&columns%5B9%5D%5Bsearchable%5D=true&columns%5B9%5D%5Borderable%5D=false&columns%5B9%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B9%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B10%5D%5Bdata%5D=compoundproperty.rd_psa&columns%5B10%5D%5Bname%5D=&columns%5B10%5D%5Bsearchable%5D=true&columns%5B10%5D%5Borderable%5D=false&columns%5B10%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B10%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B11%5D%5Bdata%5D=compoundproperty.rd_num_rotatable_bonds&columns%5B11%5D%5Bname%5D=&columns%5B11%5D%5Bsearchable%5D=true&columns%5B11%5D%5Borderable%5D=false&columns%5B11%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B11%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B12%5D%5Bdata%5D=compoundproperty.rd_num_ring_systems&columns%5B12%5D%5Bname%5D=&columns%5B12%5D%5Bsearchable%5D=true&columns%5B12%5D%5Borderable%5D=false&columns%5B12%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B12%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B13%5D%5Bdata%5D=compoundproperty.aromatic_rings&columns%5B13%5D%5Bname%5D=&columns%5B13%5D%5Bsearchable%5D=true&columns%5B13%5D%5Borderable%5D=false&columns%5B13%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B13%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B14%5D%5Bdata%5D=compoundproperty.rd_heavy_atoms&columns%5B14%5D%5Bname%5D=&columns%5B14%5D%5Bsearchable%5D=true&columns%5B14%5D%5Borderable%5D=false&columns%5B14%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B14%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B15%5D%5Bdata%5D=compoundproperty.qed_weighted&columns%5B15%5D%5Bname%5D=&columns%5B15%5D%5Bsearchable%5D=true&columns%5B15%5D%5Borderable%5D=false&columns%5B15%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B15%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B16%5D%5Bdata%5D=&columns%5B16%5D%5Bname%5D=&columns%5B16%5D%5Bsearchable%5D=true&columns%5B16%5D%5Borderable%5D=false&columns%5B16%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B16%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B17%5D%5Bdata%5D=&columns%5B17%5D%5Bname%5D=&columns%5B17%5D%5Bsearchable%5D=true&columns%5B17%5D%5Borderable%5D=false&columns%5B17%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B17%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B18%5D%5Bdata%5D=&columns%5B18%5D%5Bname%5D=&columns%5B18%5D%5Bsearchable%5D=true&columns%5B18%5D%5Borderable%5D=false&columns%5B18%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B18%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B19%5D%5Bdata%5D=ppiactivities%5B0%5D.experimental_value&columns%5B19%5D%5Bname%5D=&columns%5B19%5D%5Bsearchable%5D=true&columns%5B19%5D%5Borderable%5D=false&columns%5B19%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B19%5D%5Bsearch%5D%5Bregex%5D=false&columns%5B20%5D%5Bdata%5D=similarity&columns%5B20%5D%5Bname%5D=&columns%5B20%5D%5Bsearchable%5D=true&columns%5B20%5D%5Borderable%5D=false&columns%5B20%5D%5Bsearch%5D%5Bvalue%5D=&columns%5B20%5D%5Bsearch%5D%5Bregex%5D=false&start={}&length={}&search%5Bvalue%5D=&search%5Bregex%5D=false&is_on_page_load=false&data_source=PPI&target_columns=CUR&rd_mw_freebase_gte=74&rd_mw_freebase_lte=500&cdk_xlogp_gte=-35&cdk_xlogp_lte=5&rd_hba_gte=0&rd_hba_lte=10&rd_hbd_gte=0&rd_hbd_lte=5&rd_psa_gte=0&rd_psa_lte=1763&rd_num_rotatable_bonds_gte=0&rd_num_rotatable_bonds_lte=132&rd_num_ring_systems_gte=0&rd_num_ring_systems_lte=13&contains_null%5B%5D=compoundproperty__rd_mw_freebase&contains_null%5B%5D=compoundproperty__cdk_xlogp&contains_null%5B%5D=compoundproperty__rd_hba&contains_null%5B%5D=compoundproperty__rd_hbd&contains_null%5B%5D=compoundproperty__rd_psa&contains_null%5B%5D=compoundproperty__rd_num_rotatable_bonds&contains_null%5B%5D=compoundproperty__rd_num_ring_systems&_=1699870683256".format(start_id, length)

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the content of the request to a file
        return response.json()
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None

def get_ppi_ids_from_data(data):
    ppi_ids = []
    
    compounds = data["data"]

    for compound in compounds:
        ppi_ids.append(compound["ppi_id"])


    return ppi_ids


def load_data(filename):
    if os.path.exists(filename) and os.path.getsize(filename) != 0:
        with open(filename, "r") as file:
            json_data = json.load(file)
        return json_data
    else:
        return {"rule_of_5_compound_ids": []}

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


def save_ppi_ids_rule_of_five_to_json_file(filename):
    length = 50
    total_ppi_ids = load_data(filename)

    batch_last_time = time.time()
    for x in range(0, 9913, length):
        data = get_data(x, length)
        ppi_ids = get_ppi_ids_from_data(data)
        total_ppi_ids["rule_of_5_compound_ids"] += ppi_ids

        eta_seconds = (time.time() - batch_last_time) * (9913 - x) / length
        eta_msg = format_seconds(eta_seconds)
        batch_last_time = time.time()
        print("Downloading {} out of {}. ETA is {}".format(x, 9913, eta_msg))

        # Save to json file
        with open(filename, "w") as file:
            total_ppi_ids["rule_of_5_compound_ids"] = list(set(list(total_ppi_ids["rule_of_5_compound_ids"])))
            json.dump(total_ppi_ids, file)

if __name__ == "__main__":
    save_ppi_ids_rule_of_five_to_json_file("DLiP_PPI_DB_rule_of_5_compound_ids.json")

