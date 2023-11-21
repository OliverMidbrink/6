import json
import os
import random
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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



def create_csr_graph(edge_list):
    # Flatten the list of edges and determine unique nodes
    nodes = set([node for edge in edge_list for node in edge])
    # Create a mapping from node names to integers
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Initialize lists to store the CSR data
    row_indices = []
    col_indices = []
    data = []
    
    # Populate the lists with edge data
    for edge in edge_list:
        row_indices.append(node_to_idx[edge[0]])
        col_indices.append(node_to_idx[edge[1]])
        data.append(1)  # Assuming an unweighted graph

        # For undirected graphs, add the edge in the other direction
        row_indices.append(node_to_idx[edge[1]])
        col_indices.append(node_to_idx[edge[0]])
        data.append(1)  # Assuming an unweighted graph

    # Create the CSR matrix
    graph_csr = csr_matrix((data, (row_indices, col_indices)), shape=(len(nodes), len(nodes)))
    return graph_csr

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
    
    csr_graph = create_csr_graph(edge_list)

    print(csr_graph)
    
    plot_graph(csr_graph)

    # Fill in with equal amount of assumed non iPPI prot pairs + mol

if __name__ == "__main__":
    main()