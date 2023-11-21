from get_random_edge_sample import get_random_edge_sample
from openai import OpenAI
import requests
import sys
import pandas as pd
from get_HuRI_graph import get_neighbors_from_uniprots, get_HuRI_table_as_uniprot_edge_list

def ask_gpt(input_text, prompt, model, client):
    gpt_response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ],
        model=model,
        temperature=0.0,
        
    )
    return gpt_response.choices[0].message.content

def fetch_uniprot_data(protein_id):
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

def get_uniprot_data(uniprot_id):
    df_idmappings = pd.read_table("interactome/idmapping_2023_11_18.tsv")
    uniprot_information = df_idmappings.loc[df_idmappings["Entry"] == uniprot_id].values.tolist()
    return uniprot_information

def evaluate_edge_helpfullness(edge, instruction, client):
    iPPI_helpfullness = None

    while iPPI_helpfullness is None:
        gpt_input = str(get_uniprot_data(edge[0])) + "\n\n\nAND the second protein (could be interacting with the same protein) is\n\n\n" + str(get_uniprot_data(edge[1]))
        gpt_prompt = "From -100 to 100. 100 = should inhibit, -100 = do not inhibit. 0 for no knowledge. Just respond with the number. How helpful would inibiting the interaction between these uniprots in achiving this goal (make an educated using your comprehensive biological knowledge) (if you cant decide just enter -10 to 10): {}".format(instruction)
        gpt_evaluation = ask_gpt(gpt_input, gpt_prompt, "gpt-3.5-turbo", client)
        print(gpt_evaluation)
        try:
            iPPI_helpfullness = float(gpt_evaluation) / 100
        except:
            pass

    return iPPI_helpfullness 

def evaluate_edges(edge_list):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="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",
    )

    instruction = ""
    with open("MULTISELS/instruction.txt", "r") as file:
        instruction = file.read()

    tuples = set()
    for edge in edge_list:
        iPPI_helpfullness = evaluate_edge_helpfullness(edge, instruction, client)
        tuples.add((edge[0], edge[1], iPPI_helpfullness))

    return tuples

def main():
    interesting_uniprot_ids = ["Q01860", "Q06416", "P48431", "O43474"]

    edges_to_evaluate = get_neighbors_from_uniprots(get_HuRI_table_as_uniprot_edge_list(), interesting_uniprot_ids, n_step_neighbors=1)

    tuples = evaluate_edges(edges_to_evaluate)
    print(tuples)

if __name__ == "__main__":
    main()