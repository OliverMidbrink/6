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
import time

def get_af_uniprot_ids(af_folder="data/AlphaFoldData/"):
    uniprot_ids = {x.split("-")[1] for x in os.listdir(af_folder) if "AF-" in x}
    sorted_ids = sorted(uniprot_ids)
    print("Found {} unique Uniprot IDs in AlphaFoldData directory.".format(len(sorted_ids)))
    return sorted_ids

def ask_gpt(input_text, prompt, model, client):
    gpt_response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ],
        model=model,
        temperature=0.8,
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

def get_cost_for_one_eval(model):
    if model == "gpt-4-1106-preview":
        return 1.35 / 15 / 3
    else:
        print("Please add model pricing in the get cost function for pricing.")

def evaluate_edge_helpfullness(edge, instruction, n_rep_avg, client, model):
    iPPI_helpfullness = 0

    total_cost = 0
    reps = 0

    failed_attemps = 0
    while reps < n_rep_avg:
        try:
            gpt_input = "The first protein is\n\n\n" + str(get_uniprot_data(edge[0])) + "\n\n\nAND the second protein, which could the another protein of the same id, is:\n\n\n" + str(get_uniprot_data(edge[1])) + "\n\n\nMake your judgement and only respond with a number from -100 to 100 based on your instructions."
            gpt_prompt = "From -100 to 100. 100 = should inhibit because helps goal a lot and has no side effects, -100 = do not inhibit because does not help goal and has a lot of side effects. How helpful would inibiting the interaction between these proteins given by uniptors plus other descriptors be for achiving the following goal. Make an educated judgement using your comprehensive biological knowledge and systems understanding. The goal is as follows: {}".format(instruction)
            gpt_evaluation = ask_gpt(gpt_input, gpt_prompt, model, client)

            print(gpt_evaluation)
            total_cost += get_cost_for_one_eval(model)
        
            iPPI_helpfullness = float(gpt_evaluation) / 100.0 / float(n_rep_avg)
            reps += 1
        except Exception as e:
            failed_attemps += 1
            print(e)
            print("Attemps {}".format(failed_attemps))
            if failed_attemps > 7:
                #return 0, total_cost
                print("Attemps {}".format(failed_attemps))
            pass
    print("Cost was {} dollars".format(total_cost))
    return iPPI_helpfullness, total_cost

def evaluate_edges(edge_list, model, n_rep_avg, interesting_uniprot_ids, search_tree, file_name=None):
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
        iPPI_helpfullness, cost = evaluate_edge_helpfullness(edge, instruction, n_rep_avg, client, model)
        total_cost += cost
        tuples.add((edge[0], edge[1], iPPI_helpfullness))
        print("Total cost is {} dollars".format(total_cost))

    print("Total cost for edge list was {} dollars".format(total_cost))

    json_data = {"iPPI_tuples": list(tuples), "search_tree": search_tree, "interesting_uniprot_ids": interesting_uniprot_ids, "cost": total_cost, "model": model, "n_rep_avg": n_rep_avg, "instruction": instruction}

    if file_name is not None: # Save, will overwrite.
        with open(file_name, "w") as file:
            json.dump(json_data, file)

    return json_data

def in_edge_list(edge, edge_list):
    if edge in edge_list or [edge[1], edge[0]] in edge_list:
        return True
    return False

def both_in_uniprot_list(edge, uniprot_list): # AlphaFold
    if edge[0] in uniprot_list and edge[1] in uniprot_list:
        return True
    return False

def get_only_unique(edge_list):
    unique_pairs = list(set(tuple(sorted(pair)) for pair in edge_list))
    unique_list = [list(pair) for pair in unique_pairs]
    return unique_list

def get_eval_edges(tree_n: list, interesting_uniprot_ids: list):
    interactome_edge_list = get_HuRI_table_as_uniprot_edge_list()
    af_uniprots = get_af_uniprot_ids()
    edges_to_evaluate = []

    step = 0
    for n in tree_n:
        step += 1
        step_edges = get_neighbors_from_uniprots(interactome_edge_list, interesting_uniprot_ids, n_step_neighbors=step)
        step_edges = get_only_unique(step_edges) # Might already be unique but double check
        step_edges = [edge for edge in step_edges if both_in_uniprot_list(edge, af_uniprots)] # Filter only AF edges (because of structure data limitation)
        random.shuffle(step_edges)

        if n is not None and n < len(step_edges): # If n, sample, else take all edges
            step_edges = random.sample(step_edges, n)
        
        edges_to_evaluate += step_edges


    edges_to_evaluate = get_only_unique(edges_to_evaluate) # If graph is cyclic, filter out the duplicates
    return edges_to_evaluate

def main():
    interesting_uniprot_ids = ["Q01860", "Q06416", "P48431", "O43474"]
    search_tree = [10, 5, 3, 1]
    model = "gpt-4-1106-preview"
    n_rep_avg = 8

    edges_to_evaluate = get_eval_edges(search_tree, interesting_uniprot_ids)
    user_approval = True if input("Cost Will be projected to {} USD for this analysis. Proceed? y/n + Enter:".format(len(edges_to_evaluate) * n_rep_avg * get_cost_for_one_eval(model))) == "y" else False
    if not user_approval:
        print("Okay. Then aborting. ")
        sys.exit(0)
    print("Proceeding with analysis.")
    json = evaluate_edges(edges_to_evaluate, model=model, n_rep_avg=n_rep_avg, interesting_uniprot_ids=interesting_uniprot_ids, search_tree=search_tree, file_name="MULTISELS/latest_gpt-4_output.json")


    sys.exit(0)
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