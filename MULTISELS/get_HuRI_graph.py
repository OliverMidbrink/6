from scipy.sparse import csr_matrix
import h5py
import pandas as pd
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components
import json

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
    
    return csr_graph

def uniprot_list_from_ensg(ensg, df_idmappings):
    uniprots = list(df_idmappings.loc[df_idmappings["From"] == ensg]["Entry"])
    return uniprots

def get_HuRI_table_as_uniprot_edge_list():
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

    return edge_list

def main():
    edges = get_HuRI_table_as_uniprot_edge_list()
    json_obj = {"uniprot_edges": edges}

    with open("interactome/HuRI_uniprot_edge_list.json", "w") as file:
        json.dump(json_obj, file)

if __name__ == "__main__":
    main()