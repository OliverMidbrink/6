import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import h5py
from scipy.sparse import csr_matrix


id_mapping_path = "interactome/idmapping_2023_11_18.tsv"

# Read TSV file into DataFrame
df_id_mappings = pd.read_table(id_mapping_path)
print(df_id_mappings.head())


HuRI_path = "interactome/HuRI.tsv"

# Read TSV file into DataFrame
df_HuRI = pd.read_table(HuRI_path)
print(df_HuRI.head())

def get_full_uniprot_id_list_alpha_fold(af_folder = "data/AlphaFoldData/"):
    sorted_list = sorted(os.listdir(af_folder))
    uniprot_ids = []

    for x in sorted_list:
        if "AF-" in x:
            uniprot_ids.append(x.split("-")[1])

    sorted_ids = sorted(list(set(uniprot_ids)))
    print(len(sorted_ids))
    return sorted_ids

def to_uniprot_from_ENSG(ENSG):
    results = df_id_mappings.loc[df_id_mappings["From"] == ENSG]
    return results

def get_interactome_adjacency_matrix_from_HuRI():
    # Modfy this part for efficiency and add a progress bar. 
    # 1 create a new df for the updated HuRI data frame with new indexing (convert from ENS to UNIPROT_ID)
    # Do the reindexing and save to data/HuRI_uniprot_id.ts
    all_unique_uniprots_ids = get_full_uniprot_id_list_alpha_fold()

    adj_matrix = np.zeros((len(all_unique_uniprots_ids), len(all_unique_uniprots_ids)))
    count = 0
    for row_id in tqdm([x for x in df_HuRI.index], desc="Mapping HuRI ENSG to AlphaFold uniprot", unit="Uniprots"):
        interacting_uniprots_ids = {}
        
        for col in df_HuRI.columns:
            # change the ENSG gene id to the uniprot using the id_mappings df (second column is uniprot and first is ENSG0000 gene)
                ENSG = df_HuRI.iloc[[row_id]][col].values[0]
                interacting_uniprots_ids[col] = list(to_uniprot_from_ENSG(ENSG)["Entry"].values)

                        
        
        # Add to the ajacency matrix
        for a_uniprot in interacting_uniprots_ids['A']:
             for b_uniprot in interacting_uniprots_ids['B']:
                if a_uniprot in all_unique_uniprots_ids and b_uniprot in all_unique_uniprots_ids:
                    a_index = all_unique_uniprots_ids.index(a_uniprot)
                    b_index = all_unique_uniprots_ids.index(b_uniprot)

                    adj_matrix[a_index][b_index] = 1
                    adj_matrix[b_index][a_index] = 1
                    #print("Found {} and {} in alphafold".format(a_uniprot, b_uniprot))
                    count += 1
                else:
                    #print("Could not find {} in alphafold dataset".format(a_uniprot, b_uniprot))
                    pass
    print("Found {} interactions".format(count))
    return adj_matrix

adj_matrix = get_interactome_adjacency_matrix_from_HuRI()

print(adj_matrix)

with h5py.File('interactome/HuRI_to_Alphafold_PPI_adj_matrix.h5', 'w') as file:
    dataset = file.create_dataset('HuRI_to_Alphafold_PPI_adj_matrix', data=adj_matrix)

csr_matrix = csr_matrix(adj_matrix)

with h5py.File('interactome/HuRI_to_Alphafold_PPI_csr_matrix.h5', 'w') as file:
    # Create a dataset and save the CSR matrix
    dataset = file.create_dataset('csr_matrix', data=csr_matrix.data, shape=csr_matrix.shape, dtype='f')
    dataset.attrs['indices'] = csr_matrix.indices
    dataset.attrs['indptr'] = csr_matrix.indptr
