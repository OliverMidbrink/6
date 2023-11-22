import json
import os
import random
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components
from itertools import combinations_with_replacement
from tqdm import tqdm
import h5py
from spektral.data import Graph
from scipy.sparse import block_diag, hstack, vstack
import pandas as pd

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

def get_af_uniprot_ids(af_folder="data/AlphaFoldData/"):
    uniprot_ids = {x.split("-")[1] for x in os.listdir(af_folder) if "AF-" in x}
    sorted_ids = sorted(uniprot_ids)
    print(len(sorted_ids))
    return sorted_ids

def save_graph_to_hdf5(graph, file_path):
    csr_adjacency = graph.a.tocsr() if not isinstance(graph.a, csr_matrix) else graph.a

    with h5py.File(file_path, 'w') as f:
        # Save the features matrix
        f.create_dataset('features', data=graph.x)
        # Save the adjacency matrix in CSR format
        f.create_dataset('data', data=csr_adjacency.data)
        f.create_dataset('indices', data=csr_adjacency.indices)
        f.create_dataset('indptr', data=csr_adjacency.indptr)
        f.create_dataset('shape', data=csr_adjacency.shape)
        # Save the labels or targets if they exist
        if hasattr(graph, 'y') and graph.y is not None:
            f.create_dataset('labels', data=graph.y)

def load_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        # Load the features matrix
        features = f['features'][:]
        # Load the CSR components
        data = f['data'][:]
        indices = f['indices'][:]
        indptr = f['indptr'][:]
        shape = f['shape'][:]
        # Create the CSR matrix
        csr_graph = csr_matrix((data, indices, indptr), shape=shape)
    
    return csr_graph, features

def is_in_DLiP(uniprot_pair_list, smiles, DLiP_data):
    for DLiP_value in DLiP_data.values():
        if set(DLiP_value["proteins"]) == set(uniprot_pair_list): # If PPI pair is in DLiP, skip
            return True
    return False

def in_HuRI(uniprot_pair_list, edge_list):
    if uniprot_pair_list in edge_list or [uniprot_pair_list[1], uniprot_pair_list[0]] in edge_list:
        return True
    else:
        return False
        print("not in HuRI")

def get_non_iPPIs(n, DLiP_keys, DLiP_data):
    # Uniprots and smiles from only the DLiP_keys
    uniprots = get_uniprots(DLiP_data.keys(), DLiP_data)
    af_uniprots = get_af_uniprot_ids()
    smiles = get_smiles(DLiP_data.keys(), DLiP_data)

    all_combs = list(combinations_with_replacement(uniprots, 2))
    random.shuffle(all_combs)

    edge_list = get_HuRI_table_as_uniprot_edge_list()

    non_iPPIs = set()

    for uniprot_pair in all_combs:
        uniprot_pair_list = list(uniprot_pair)
        # Get a random uniprot_pair (including homo pairs)
        # Get a smiles
        random_smiles = random.choice(list(smiles))

        if uniprot_pair_list[0] not in af_uniprots or uniprot_pair_list[1] not in af_uniprots:
            continue

        if in_HuRI(uniprot_pair_list, edge_list): # If it is a known interaction
            if not is_in_DLiP(uniprot_pair_list, smiles, DLiP_data): # Assume it is not an iPPI and add
                non_iPPI = (uniprot_pair_list[0], uniprot_pair_list[1], random_smiles)
                non_iPPIs.add(non_iPPI)

                if len(non_iPPIs) == n:
                    break      
  
    return non_iPPIs

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

def get_smiles(DLiP_keys, DLiP_data):
    smiles = set()
    for key in DLiP_keys:
        smiles.add(DLiP_data[key]["SMILES"][0])
    return smiles

def get_uniprots(keys, DLiP_data):
    proteins = set()
    for key in keys:
        for prot in DLiP_data[key]["proteins"]:
            proteins.add(prot)
    return proteins

def main():
    af_uniprots = get_af_uniprot_ids()
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
    print(set(train_uniprots) & set(val_uniprots) & set(test_uniprots)) # Check for overlapping prot ids

    # DLiP ids are successfully split if program returns set(), 7021, set()
    # Inflate datasets with equal amount of non interacting

    # Assume if a molecule is not associated with a pair, it does not inhibit that pair.[Assumption] to create negative training data
    # Fill in with equal amount of assumed non iPPI prot pairs + mol
    train_PPI_molecules = {}
    id_count = 0
    added = 0
    for DLiP_id in train_DLiP_ids:
        # Positive iPPI, molecule inhibits interaction
        if DLiP_data[DLiP_id]["proteins"][0] in af_uniprots and DLiP_data[DLiP_id]["proteins"][1] in af_uniprots:
            iPPI = {"proteins": DLiP_data[DLiP_id]["proteins"], "molecule": DLiP_data[DLiP_id]["SMILES"][0], "iPPI": 1} # 0 for the canonical RdKit smiles
            train_PPI_molecules[id_count] = iPPI
            id_count += 2
            added += 1
    
    id_count = 1
    for non_iPPI in get_non_iPPIs(added, train_DLiP_ids, DLiP_data):
        iPPI = {"proteins": [non_iPPI[0], non_iPPI[1]], "molecule": non_iPPI[2], "iPPI": 0} # 0 for the canonical RdKit smiles
        train_PPI_molecules[id_count] = iPPI
        id_count += 2
    
    
    val_PPI_molecules = {}
    id_count = 0
    added = 0
    for DLiP_id in val_DLiP_ids:
        # Positive iPPI, molecule inhibits interaction
        if DLiP_data[DLiP_id]["proteins"][0] in af_uniprots and DLiP_data[DLiP_id]["proteins"][1] in af_uniprots:
            iPPI = {"proteins": DLiP_data[DLiP_id]["proteins"], "molecule": DLiP_data[DLiP_id]["SMILES"][0], "iPPI": 1} # 0 for the canonical RdKit smiles
            val_PPI_molecules[id_count] = iPPI
            id_count += 2
            added += 1
    
    id_count = 1
    for non_iPPI in get_non_iPPIs(added, val_DLiP_ids, DLiP_data):
        iPPI = {"proteins": [non_iPPI[0], non_iPPI[1]], "molecule": non_iPPI[2], "iPPI": 0} # 0 for the canonical RdKit smiles
        val_PPI_molecules[id_count] = iPPI
        id_count += 2

    
    test_PPI_molecules = {}
    id_count = 0
    added = 0
    for DLiP_id in test_DLiP_ids:
        # Positive iPPI, molecule inhibits interaction
        if DLiP_data[DLiP_id]["proteins"][0] in af_uniprots and DLiP_data[DLiP_id]["proteins"][1] in af_uniprots:
            iPPI = {"proteins": DLiP_data[DLiP_id]["proteins"], "molecule": DLiP_data[DLiP_id]["SMILES"][0], "iPPI": 1} # 0 for the canonical RdKit smiles
            test_PPI_molecules[id_count] = iPPI
            id_count += 2
            added += 1
    
    id_count = 1
    for non_iPPI in get_non_iPPIs(added, test_DLiP_ids, DLiP_data):
        iPPI = {"proteins": [non_iPPI[0], non_iPPI[1]], "molecule": non_iPPI[2], "iPPI": 0} # 0 for the canonical RdKit smiles
        test_PPI_molecules[id_count] = iPPI
        id_count += 2
    

    # Now we have the following dicts
    # train_PPI_molecules
    # val_PPI_molecules
    # test_PPI_molecules

    train_path = "data/iPPI_graphs/train_graphs"
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    skipped = 0
    for key in tqdm(sorted(list(train_PPI_molecules.keys())), desc="Saving training data", unit="c_graphs"):
        try:
            iPPI = train_PPI_molecules[key]
            file_name = os.path.join(train_path, '{}_protA_{}_protB_{}_smiles_{}_is_iPPI_{}.hdf5'.format(key, iPPI['proteins'][0], iPPI['proteins'][1], iPPI['molecule'], iPPI["iPPI"]))

            csr_protA, feat_protA = load_from_hdf5('data/protein_atom_graphs/AF-{}-F1-model_v4_graph.hdf5'.format(iPPI['proteins'][0]))
            csr_protB, feat_protB = load_from_hdf5('data/protein_atom_graphs/AF-{}-F1-model_v4_graph.hdf5'.format(iPPI['proteins'][1]))
            csr_mol, feat_mol = load_from_hdf5('data/mol_graphs/{}_graph.hdf5'.format(iPPI['molecule']))

            # Combine the adjacency matrices
            combined_adjacency = block_diag((csr_protA, csr_protB, csr_mol))

            # Combine the feature matrices
            combined_features = np.vstack((feat_protA, feat_protB, feat_mol))

            # Create a Spektral Graph object
            graph = Graph(x=combined_features, a=combined_adjacency, y=np.array([iPPI["iPPI"]], dtype=np.float32))

            save_graph_to_hdf5(graph, file_name)
        except:
            skipped += 1
    
    print("Skipped {} files.".format(skipped))



    val_path = "data/iPPI_graphs/val_graphs"
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    skipped = 0
    for key in tqdm(sorted(list(val_PPI_molecules.keys())), desc="Saving validation data", unit="c_graphs"):
        try:
            iPPI = val_PPI_molecules[key]
            file_name = os.path.join(val_path, '{}_protA_{}_protB_{}_smiles_{}_is_iPPI_{}.hdf5'.format(key, iPPI['proteins'][0], iPPI['proteins'][1], iPPI['molecule'], iPPI["iPPI"]))

            csr_protA, feat_protA = load_from_hdf5('data/protein_atom_graphs/AF-{}-F1-model_v4_graph.hdf5'.format(iPPI['proteins'][0]))
            csr_protB, feat_protB = load_from_hdf5('data/protein_atom_graphs/AF-{}-F1-model_v4_graph.hdf5'.format(iPPI['proteins'][1]))
            csr_mol, feat_mol = load_from_hdf5('data/mol_graphs/{}_graph.hdf5'.format(iPPI['molecule']))

            # Combine the adjacency matrices
            combined_adjacency = block_diag((csr_protA, csr_protB, csr_mol))

            # Combine the feature matrices
            combined_features = np.vstack((feat_protA, feat_protB, feat_mol))

            # Create a Spektral Graph object
            graph = Graph(x=combined_features, a=combined_adjacency, y=np.array([iPPI["iPPI"]], dtype=np.float32))

            save_graph_to_hdf5(graph, file_name)
        except:
            skipped += 1

    print("Skipped {} files.".format(skipped))

    test_path = "data/iPPI_graphs/test_graphs"
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    skipped = 0
    for key in tqdm(sorted(list(test_PPI_molecules.keys())), desc="Saving test data", unit="c_graphs"):
        try:
            iPPI = test_PPI_molecules[key]
            file_name = os.path.join(test_path, '{}_protA_{}_protB_{}_smiles_{}_is_iPPI_{}.hdf5'.format(key, iPPI['proteins'][0], iPPI['proteins'][1], iPPI['molecule'], iPPI["iPPI"]))

            csr_protA, feat_protA = load_from_hdf5('data/protein_atom_graphs/AF-{}-F1-model_v4_graph.hdf5'.format(iPPI['proteins'][0]))
            csr_protB, feat_protB = load_from_hdf5('data/protein_atom_graphs/AF-{}-F1-model_v4_graph.hdf5'.format(iPPI['proteins'][1]))
            csr_mol, feat_mol = load_from_hdf5('data/mol_graphs/{}_graph.hdf5'.format(iPPI['molecule']))

            # Combine the adjacency matrices
            combined_adjacency = block_diag((csr_protA, csr_protB, csr_mol))

            # Combine the feature matrices
            combined_features = np.vstack((feat_protA, feat_protB, feat_mol))

            # Create a Spektral Graph object
            graph = Graph(x=combined_features, a=combined_adjacency, y=np.array([iPPI["iPPI"]], dtype=np.float32))

            save_graph_to_hdf5(graph, file_name)
        except:
            skipped += 1
    
    print("Skipped {} files.".format(skipped))

if __name__ == "__main__":
    main()