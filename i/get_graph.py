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
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    numeric_edges = np.array([(node_mapping[u], node_mapping[v]) for u, v in G.edges()])

    # Extract node and edge features
    node_features = np.array([G.nodes[node]['features'] for node in G.nodes()])
    edge_features = np.array([G.edges[edge]['features'] for edge in G.edges()])

    # Create adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(G, nodelist=node_mapping.keys())
    print(adjacency_matrix)

    # Create Spektral graph
    graph = spektral.data.Graph(x=node_features, a=adjacency_matrix, e=edge_features, edges=numeric_edges)
    return graph

def get_graph(edge_list):
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # Assign random features to nodes
    mean, std_dev = 0, 1  # Mean and standard deviation for normal distribution

    for node in G.nodes():
        G.nodes[node]['features'] = np.random.normal(mean, std_dev, 2)

    # Assign random features to edges
    for edge in G.edges():
        G.edges[edge]['features'] = np.random.normal(mean, std_dev, 2)

    return G

def get_edgelist():
    json_data = {}
    with open("i/HuRI_edge_list.json", "r") as file:
        json_data = json.load(file)
    return json_data["HuRI_edge_list"]



ensg_edges = get_HuRI_table_as_edge_list()

graph = get_graph(ensg_edges)

graph_next = update(graph)

from_nx_to_spektral(graph)