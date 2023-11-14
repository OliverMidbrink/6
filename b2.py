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


def no_stream_time():
    start_time = time.time()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
    )
    time_taken = time.time() - start_time
    print("took {} seconds. ".format(time.time() - start_time))
    print(chat_completion.choices[0].message.content)
    return time_taken




def stream_time():
    start_time = time.time()
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say this is a test"}],
        stream=True,
    )
    for part in stream:
        print(part.choices[0].delta.content or "", end="")
    print("")
    time_taken = time.time() - start_time

    print("took {} seconds. ".format(time.time() - start_time))
    return time_taken


class RCT_stream_no_stream():
    def __init__(self, methods):
        print("init")
        self.methods = methods

    def average(self, list):
        return np.array(list).mean()


    def run_analysis(self, n):
        run_data = {}
        for method in self.methods:
            run_data[method.__name__] = {}
            run_data[method.__name__]["run_times"] = []

        for x in range(n):
            method_choice = random.choice(self.methods)
            print("Starting {} method. Run {} out of {}".format(method_choice.__name__, x, n))
            run_time = method_choice()
            print("Took {} seconds".format(run_time))

            run_data[method_choice.__name__]["run_times"].append(run_time)

        print(run_data)

        print("\n"*5)
        print("Run complete.")
        print("=" * 20)

        with open("RCT_experiment_{}.json".format(random.randint(0, 100000000000)), 'w') as file:
            json.dump(run_data, file)
        for method in self.methods:
            print("Method {} took in average {} seconds".format(method.__name__, self.average(run_data[method.__name__]["run_times"])))
        

RCT_stream_no_stream([stream_time, no_stream_time]).run_analysis(100)

sys.exit(0)

def ask_gpt(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content


def ask_gpt_stream(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
        stream=True,
    )
    return chat_completion.choices[0].message.content


stream = ask_gpt_stream("tessdflj")
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
