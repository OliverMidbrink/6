from get_random_edge_sample import get_random_edge_sample
from openai import OpenAI
import requests
import sys
import pandas as pd
from get_HuRI_graph import get_neighbors_from_uniprots, get_HuRI_table_as_uniprot_edge_list
import json
from tqdm import tqdm
import random
import os

def get_af_uniprot_ids(af_folder="data/AlphaFoldData/"):
    uniprot_ids = {x.split("-")[1] for x in os.listdir(af_folder) if "AF-" in x}
    sorted_ids = sorted(uniprot_ids)
    print(len(sorted_ids))
    return sorted_ids

def ask_gpt(input_text, prompt, model, client):
    gpt_response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ],
        model=model,
        temperature=0.0,
        timeout=10,
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

def get_cost(input_chars, output_chars, model):
    input_tokens = input_chars / 4
    output_tokens = output_chars / 4
    if model == "gpt-3.5-turbo-1106":
        return 0.001 / 1000 * input_tokens + 0.002 / 1000 * output_tokens
    if model == "gpt-4-1106-preview":
        return 0.01 / 1000 * input_tokens + 0.03 / 1000 * output_tokens
    else:
        print("Please add model pricing in the get cost function for pricing.")

def evaluate_edge_helpfullness(edge, instruction, client, model):
    iPPI_helpfullness = None

    total_cost = 0

    failed_attemps = 0
    while iPPI_helpfullness is None:
        try:
            gpt_input = str(get_uniprot_data(edge[0])) + "\n\n\nAND the second protein (could be interacting with the same protein) is\n\n\n" + str(get_uniprot_data(edge[1])) + "Only answer with a number -100 to 100."
            gpt_prompt = "From -100 to 100. 100 = should inhibit, -100 = do not inhibit. 0 for no knowledge. Just respond with the number. How helpful would inibiting the interaction between these uniprots in achiving this goal (make an educated using your comprehensive biological knowledge) (if you cant decide just enter -10 to 10): {}".format(instruction)
            total_input_len = len(gpt_input + gpt_prompt)
            gpt_evaluation = ask_gpt(gpt_input, gpt_prompt, model, client)
            output_len = len(gpt_evaluation)

            total_cost += get_cost(total_input_len, output_len, model)
            print(gpt_evaluation)
        
            iPPI_helpfullness = float(gpt_evaluation) / 100
        except:
            failed_attemps += 1
            print("Attemps {}".format(failed_attemps))
            if failed_attemps > 7:
                #return 0, total_cost
                print("Attemps {}".format(failed_attemps))
            pass
    print("Cost was {} dollars".format(total_cost))
    return iPPI_helpfullness, total_cost

def evaluate_edges(edge_list, model):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="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",
    )

    instruction = ""
    with open("MULTISELS/instruction.txt", "r") as file:
        instruction = file.read()

    total_cost = 0
    tuples = set()

    for edge in tqdm(edge_list, desc="Evaluating protein interactions", unit="iPPIs"):
        iPPI_helpfullness, cost = evaluate_edge_helpfullness(edge, instruction, client, model)
        total_cost += cost
        tuples.add((edge[0], edge[1], iPPI_helpfullness))
        print("Total cost is {} dollars".format(total_cost))

    print("Total cost for edge list was {} dollars".format(total_cost))
    return list(tuples)

def get_eval_edges(tree_n: list, interesting_uniprot_ids: list):
    interactome_edge_list = get_HuRI_table_as_uniprot_edge_list()

    step = 0
    for n in tree_n:
        step += 1
        step_edges = get_neighbors_from_uniprots(interactome_edge_list, interesting_uniprot_ids, n_step_neighbors=step)
        random.shuffle(step_edges)
        
        


def main():
    interesting_uniprot_ids = ["Q01860", "Q06416", "P48431", "O43474"]
    af_uniprots = get_af_uniprot_ids()

    two_step_neighbors = 
    n_eval_two_step = 3

    edges_to_evaluate = []
    while len(edges_to_evaluate) < n_eval_two_step:
        edge = random.choice(two_step_neighbors)
        if edge[0] in af_uniprots and edge[1] in af_uniprots:
            if edge not in edges_to_evaluate and [edge[1], edge[0]] not in edges_to_evaluate:
                edges_to_evaluate.append(edge)
        
    print("{} edges to evaluate".format(len(edges_to_evaluate)))
    
    tuples = evaluate_edges(edges_to_evaluate, model="gpt-4-1106-preview", n_rep_avg=3) # Use GPT4 and 3 repetition average

    with open("MULTISELS/iPPI_eval_GPT4_NREPAVG_3.json", "w") as file:
        json.dump({"tuples": tuples}, file)

    """
    os.makedirs("MULTISELS/BENCHMARKING")
    ## Create comparison
    tuples_1 = evaluate_edges(edges_to_evaluate, model="gpt-3.5-turbo-1106")
    tuples_2 = evaluate_edges(edges_to_evaluate, model="gpt-3.5-turbo-1106")
    tuples_3 = evaluate_edges(edges_to_evaluate, model="gpt-3.5-turbo-1106")

    tuples_1_gpt_4 = evaluate_edges(edges_to_evaluate, model="gpt-4-1106-preview")
    tuples_2_gpt_4 = evaluate_edges(edges_to_evaluate, model="gpt-4-1106-preview")
    tuples_3_gpt_4 = evaluate_edges(edges_to_evaluate, model="gpt-4-1106-preview")     

    with open("MULTISELS/BENCHMARKING/com1.json", "w") as file:
        json_data = {"tuples": tuples_1}
        json.dump(json_data, file)
    with open("MULTISELS/BENCHMARKING/com2.json", "w") as file:
        json_data = {"tuples": tuples_2}
        json.dump(json_data, file)
    with open("MULTISELS/BENCHMARKING/com3.json", "w") as file:
        json_data = {"tuples": tuples_3}
        json.dump(json_data, file)


    with open("MULTISELS/BENCHMARKING/com1_gpt4.json", "w") as file:
        json_data = {"tuples": tuples_1_gpt_4}
        json.dump(json_data, file)
    with open("MULTISELS/BENCHMARKING/com2_gpt4.json", "w") as file:
        json_data = {"tuples": tuples_2_gpt_4}
        json.dump(json_data, file)
    with open("MULTISELS/BENCHMARKING/com3_gpt4.json", "w") as file:
        json_data = {"tuples": tuples_3_gpt_4}
        json.dump(json_data, file)
    

    with open("MULTISELS/BENCHMARKING/com1.json", "r") as file:
        tuples_1 = json.load(file)
    with open("MULTISELS/BENCHMARKING/com2.json", "r") as file:
        tuples_2 = json.load(file)
    with open("MULTISELS/BENCHMARKING/com3.json", "r") as file:
        tuples_3 = json.load(file)


    with open("MULTISELS/BENCHMARKING/com1_gpt4.json", "r") as file:
        tuples_1_gpt_4 = json.load(file)
    with open("MULTISELS/BENCHMARKING/com2_gpt4.json", "r") as file:
        tuples_2_gpt_4 = json.load(file)
    with open("MULTISELS/BENCHMARKING/com3_gpt4.json", "r") as file:
        tuples_3_gpt_4 = json.load(file)
    """


if __name__ == "__main__":
    main()