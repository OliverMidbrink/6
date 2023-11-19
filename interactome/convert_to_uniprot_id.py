import pandas as pd


id_mapping_path = "data/idmapping_2023_11_18.tsv"

# Read TSV file into DataFrame
df_id_mappings = pd.read_table(id_mapping_path)
print(df_id_mappings.head())


HuRI_path = "data/HuRI.tsv"

# Read TSV file into DataFrame
df_HuRI = pd.read_table(HuRI_path)
print(df_HuRI.head())


def to_uniprot_from_ENSG(ENSG):
    results = df_id_mappings.loc[df_id_mappings["From"] == ENSG]
    return results

def convert_HuRI_dot_tsv_to_uniprot():
    # Modfy this part for efficiency and add a progress bar. 
    # 1 create a new df for the updated HuRI data frame with new indexing (convert from ENS to UNIPROT_ID)
    # Do the reindexing and save to data/HuRI_uniprot_id.ts
    print()

    for row_id in [x for x in df_HuRI.index]:
        for col in df_HuRI.columns:
            # change the ENSG gene id to the uniprot using the id_mappings df (second column is uniprot and first is ENSG0000 gene)
                ENSG = df_HuRI.iloc[[row_id]][col].values[0]

                uniprot_id = to_uniprot_from_ENSG(ENSG)
                print(uniprot_id)
            # Get t
    
# Thank you.

convert_HuRI_dot_tsv_to_uniprot()