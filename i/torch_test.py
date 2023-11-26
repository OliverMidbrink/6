import torch
import numpy as np
import json
import networkx as nx
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from torch_geometric.nn import GCNConv

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

def get_HuRI_graph():
    edge_list = get_HuRI_table_as_edge_list()

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

def to_pytorch_from_nx(G):
    # Convert node features to a tensor
    node_features = torch.tensor(np.array([G.nodes[node]['features'] for node in G.nodes()]))

    # Convert edges to a tensor (edge index)
    edge_list = [(u, v) for u, v in G.edges()]

    # Create a mapping from node IDs (strings) to unique integers
    unique_nodes = set(node for edge in edge_list for node in edge)
    node_mapping = {node: i for i, node in enumerate(unique_nodes)}

    # Convert edge list to numeric indices
    numeric_edge_list = [(node_mapping[node1], node_mapping[node2]) for node1, node2 in edge_list]

    edge_index = torch.tensor(numeric_edge_list, dtype=torch.long).t().contiguous()

    # If your graph has edge features
    edge_features = torch.tensor(np.array([G[u][v]['features'] for u, v in G.edges()]))

    #print("Node Features:\n", node_features)
    #print("Edge Index:\n", edge_index)
    #print("Edge Features:\n", edge_features)

    return node_features, edge_index, edge_features


def main():
    graph = get_HuRI_graph()
    pytorch_tensor_graph = to_pytorch_from_nx(graph)

if __name__ == "__main__":
    main()