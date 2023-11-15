import requests
import zipfile
import os
from tqdm import tqdm

def download_file(url, filename):
    """
    Download a file with a progress bar.

    Args:
    url (str): The URL of the file to download.
    filename (str): The filename under which the file will be saved.
    """
    with requests.get(url, stream=True) as response:
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(filename, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

def extract_specific_files(filename, extract_to='.', keyword='Homo_sapiens'):
    """
    Extract only files containing a specific keyword in the filename from a ZIP file.

    Args:
    filename (str): The filename of the zip file.
    keyword (str): Keyword to search for in file names.
    extract_to (str): The directory to extract to.
    """
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        # Extract only files that contain the keyword
        for file in zip_ref.namelist():
            if keyword in file:
                zip_ref.extract(file, extract_to)
                print(f"Extracted {file} to {extract_to}")

def download_and_extract(url, extract_to='./data/'):
    """
    Download a ZIP file from a URL and extract its contents.

    Args:
    url (str): The URL of the file to download.
    extract_to (str): Directory where contents are extracted.
    """

    # Extract the filename from the URL
    filename = url.split('/')[-1]

    # Download the file with a progress bar
    print(f"Downloading {filename}...")
    download_file(url, filename)
    print(f"Downloaded {filename}.")

    # Extract specific files from the zip file
    extract_specific_files(filename, extract_to)

    # Optionally, remove the zip file after extraction
    os.remove(filename)
    print(f"Removed {filename}.")

# URL of the file to be downloaded
url = "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.227/BIOGRID-ORGANISM-4.4.227.tab3.zip"

# Download and extract the file
download_and_extract(url)
