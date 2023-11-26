import pandas as pd
import gzip
import json

# Path to your GTF file
gtf_file = 'data/Homo_sapiens.GRCh38.110.gtf.gz'

# Define the column names (GTF format specification)
col_names = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

# Read the GTF file
with gzip.open(gtf_file, 'rt') as f:
    # Read the file into a DataFrame
    # GTF files are tab-delimited; comments and headers start with '#'
    df = pd.read_csv(f, comment='#', sep='\t', names=col_names)

# Extract the ENSG strings
# The 'attribute' field contains the gene_id (ENSG) among other information
df['gene_id'] = df['attribute'].str.extract('gene_id "([^"]+)"')

# Display the first few ENSG strings
ENSG_strings = list(df['gene_id'])
print(len(ENSG_strings))

with open("i/ENSG_list_full.json", "w") as f:
    json_data = {"ENSG_strings": ENSG_strings}
    json.dump(json_data, f)