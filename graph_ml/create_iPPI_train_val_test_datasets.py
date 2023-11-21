import json
import os
import random
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components
from itertools import combinations_with_replacement

def get_DLiP_ids_from_nodes(train_nodes, DLiP_data):
    DLiP_ids = set()

    for uniprot_PPI_set in train_nodes:
        for uniprot_pair in list(combinations_with_replacement(uniprot_PPI_set, 2)):
            # Check if pair is in DLiP and add in that case
            for value in DLiP_data.values():
                pair = value["proteins"]
                if set(pair) == set(uniprot_pair):
                    DLiP_ids.add(value["compound_id"]) # incorrectly labeled as compound id but it is actually DLiP id
    
    return DLiP_ids

def plot_graph(G):
    # The graph to visualize
    G = nx.from_scipy_sparse_array(G)

    # 3d spring layout
    pos = nx.spring_layout(G, dim=3, seed=779)
    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, ec="w")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    _format_axes(ax)
    fig.tight_layout()
    plt.show()

def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def get_subgraph_nodes_from_edgelist(edge_list):
    # Create a set of all nodes
    nodes = set(node for edge in edge_list for node in edge)
    # Create a mapping from node names to integers
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    # Inverse mapping to retrieve node names from indices
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Initialize lists to store the edges in terms of indices
    data = []
    rows = []
    cols = []
    
    # Populate the edge index lists
    for edge in edge_list:
        src, dst = edge
        rows.append(node_to_idx[src])
        cols.append(node_to_idx[dst])
        data.append(1)  # Assuming unweighted graph, use 1 as the placeholder for edge existence
    
    # Number of nodes
    n_nodes = len(nodes)
    # Create the CSR matrix
    csr_graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    
    # Find the connected components
    n_components, labels = connected_components(csgraph=csr_graph, directed=False, return_labels=True)
    
    # Group node indices by their component label
    subgraphs = {i: set() for i in range(n_components)}
    for node_idx, component_label in enumerate(labels):
        subgraphs[component_label].add(idx_to_node[node_idx])
    
    # Convert the dictionary to a list of sets of node names
    subgraph_node_sets = list(subgraphs.values())
    
    return subgraph_node_sets

def get_uniprots(keys, DLiP_data):
    proteins = set()
    for key in keys:
        for prot in DLiP_data[key]["proteins"]:
            proteins.add(prot)
    return proteins

def main():
    output_dir_path = "data/mol_graphs"
    molecule_smiles = "DLiP_rule_of_5_compound_data.json"
    DLiP_data = {}

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    with open(molecule_smiles, "r") as file:
        DLiP_data = json.load(file)
    
    edge_list = [x["proteins"] for x in DLiP_data.values()]
    
    # Create a graph of all the interactions. And every cluster (interacting proteins) create a set of the uniprot_ids
    sub_graph_nodes = get_subgraph_nodes_from_edgelist(edge_list)
    random.shuffle(sub_graph_nodes)

    train_val_split = int(len(sub_graph_nodes) * 0.8)
    val_test_split = int(len(sub_graph_nodes) * 0.9)

    train_nodes = sub_graph_nodes[:train_val_split]
    val_nodes = sub_graph_nodes[train_val_split:val_test_split]
    test_nodes = sub_graph_nodes[val_test_split:]

    train_DLiP_ids = get_DLiP_ids_from_nodes(train_nodes, DLiP_data)
    val_DLiP_ids = get_DLiP_ids_from_nodes(val_nodes, DLiP_data)
    test_DLiP_ids = get_DLiP_ids_from_nodes(test_nodes, DLiP_data)

    train_uniprots = get_uniprots(train_DLiP_ids, DLiP_data)
    val_uniprots = get_uniprots(val_DLiP_ids, DLiP_data)
    test_uniprots = get_uniprots(test_DLiP_ids, DLiP_data)


    print(set(train_DLiP_ids) & set(val_DLiP_ids) & set(test_DLiP_ids)) # Check for DLiP ids that are on multiple sets
    print(len(set(train_DLiP_ids) | set(val_DLiP_ids) | set(test_DLiP_ids))) # Check number of DLiP ids in total
    print(set(train_uniprots) & set(val_uniprots) & set(test_uniprots)) # Check for overlapping prot ids:

    


    # Fill in with equal amount of assumed non iPPI prot pairs + mol

if __name__ == "__main__":
    main()