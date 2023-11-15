import requests
import gzip
import os
from tqdm import tqdm

def download_file_with_progress(url, local_filename):
    """
    Download a file with a progress bar.
    """
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading")
    with open(local_filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong during download")

def extract_gz_file_with_progress(filepath, output_folder):
    """
    Extract a .gz file with a progress bar.
    """
    with gzip.open(filepath, 'rb') as f_in:
        filename = os.path.basename(filepath).replace('.gz', '')
        output_path = os.path.join(output_folder, filename)
        with open(output_path, 'wb') as f_out, tqdm(desc="Extracting", unit='iB', unit_scale=True) as progress_bar:
            while chunk := f_in.read(1024):
                f_out.write(chunk)
                progress_bar.update(len(chunk))
        print(f"Extracted: {output_path}")

# URL of the file to be downloaded
url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz"

# Local path for downloaded file
local_filename = url.split('/')[-1]
download_path = os.path.join(".", "data", local_filename)

# Ensure the ./data/ directory exists
if not os.path.exists("./data"):
    os.makedirs("./data")

# Download the file with progress bar
download_file_with_progress(url, download_path)

# Extract the file with progress bar
extract_gz_file_with_progress(download_path, "./data")

# Optionally, remove the .gz file after extraction
os.remove(download_path)
print(f"Removed downloaded .gz file: {download_path}")
