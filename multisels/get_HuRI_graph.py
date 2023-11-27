from scipy.sparse import csr_matrix
import h5py
import pandas as pd
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components
import json
from collections import defaultdict
from scipy.sparse.csgraph import breadth_first_order
import networkx as nx
import os

def get_neighbors_from_uniprots(edge_list, uniprot_ids, n_step_neighbors=3):
    # Create a graph from the edge list
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Find all neighbors for each UniProt ID up to n_neighbors away
    uniprot_edges_of_n_neighbors = []
    for uniprot_id in uniprot_ids:
        if uniprot_id in G:
            # Get all nodes within n_neighbors hops from uniprot_id
            neighbors = nx.single_source_shortest_path_length(G, uniprot_id, cutoff=n_step_neighbors)
            # Get edges between uniprot_id and its neighbors
            for neighbor, distance in neighbors.items():
                if distance > 0:  # Exclude self-loops
                    uniprot_edges_of_n_neighbors.append((uniprot_id, neighbor))
    
    return uniprot_edges_of_n_neighbors


def uniprot_list_from_ensg(ensg, df_idmappings):
    uniprots = list(df_idmappings.loc[df_idmappings["From"] == ensg]["Entry"])
    return uniprots

def get_HuRI_table_as_uniprot_edge_list():
    if os.path.exists("interactome/HuRI_edge_list.json"):
        print("Loading saved edgelist.")
        with open("interactome/HuRI_edge_list.json", "r") as file:
            return json.load(file)["HuRI_edge_list"]
    else:
        print("Generating edgelist...")
        edge_list = []
        df = pd.read_table("interactome/HuRI.tsv")
        df_idmappings = pd.read_table("interactome/idmapping_2023_11_18.tsv")

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Reading mappings"):
            interact_A = row['A']
            interact_B = row['B']

            uniprots_A = uniprot_list_from_ensg(interact_A, df_idmappings)
            uniprots_B = uniprot_list_from_ensg(interact_B, df_idmappings)

            for uniprot_A in uniprots_A:
                for uniprot_B in uniprots_B:
                    edge_list.append([uniprot_A, uniprot_B])

        with open("interactome/HuRI_edge_list.json", "w") as file:
            json_data = {"HuRI_edge_list": edge_list}
            json.dump(json_data, file)
        return get_HuRI_table_as_uniprot_edge_list()

def main():
    edges = get_HuRI_table_as_uniprot_edge_list()
    json_obj = {"uniprot_edges": edges}

    with open("interactome/HuRI_uniprot_edge_list.json", "w") as file:
        json.dump(json_obj, file)

if __name__ == "__main__":
    main()