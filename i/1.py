import json
import networkx as nx
from tqdm import tqdm
import pandas as pd
import os
import spektral
import numpy as np

def get_HuRI_table_as_edge_list():
    if os.path.exists("i/HuRI_ENSG_EDGELIST.json"):
        edge_list = []
        print("loading edgelist.")
        with open("i/HuRI_ENSG_EDGELIST.json", "r") as f:
            edge_list = json.load(f)["HuRI_ENSG_EDGES"]
        return edge_list
    else:
        print("Generating edgelist...")
        edge_list = []
        df = pd.read_table("interactome/HuRI.tsv")

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Reading mappings"):
            interact_A = row['A']
            interact_B = row['B']

            edge_list.append([interact_A, interact_B])
        with open("i/HuRI_ENSG_EDGELIST.json", "w") as f:
            json.dump({"HuRI_ENSG_EDGES": edge_list}, f)
        return edge_list

def from_nx_to_spektral(G):
    A = nx.adjacency_matrix(G)

    # Extract node features
    # Assuming each node has the same number of features
    num_nodes = G.number_of_nodes()
    features = [print(G.nodes[key]['feature']) for key in G.nodes()]
    X = np.array(features)

    # Create a Spektral graph
    graph = spektral.data.Graph(x=X, a=A)
    return graph

def get_graph(edge_list):
    G = nx.Graph()
    G.add_edges_from(edge_list)
    return G

def get_edgelist():
    json_data = {}
    with open("i/HuRI_edge_list.json", "r") as file:
        json_data = json.load(file)
    return json_data["HuRI_edge_list"]



ensg_edges = get_HuRI_table_as_edge_list()

graph = get_graph(ensg_edges)
from_nx_to_spektral(graph)