import scanpy as sc
import os
import gzip

transcriptome_data_folder = os.path.join("data", "transcriptomes")
transcriptome_files = [file for file in os.listdir(transcriptome_data_folder) if not ".tar" in file]
for file in transcriptome_files:

    # Path to the folder containing 'matrix.mtx', 'genes.tsv', and 'barcodes.tsv'
    data_10X_folder_path = os.path.join(transcriptome_data_folder, file)

    # Read the 10x dataset
    adata = sc.read_10x_mtx(data_10X_folder_path, var_names='gene_symbols', cache=True)

    # Explore the data
    print(adata)
