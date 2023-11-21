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



def main():
    csr_graph = load_csr_graph(edge_list)

    print(csr_graph)
    
    plot_graph(csr_graph)

    # Fill in with equal amount of assumed non iPPI prot pairs + mol

if __name__ == "__main__":
    main()