import requests
from tqdm import tqdm

def download_file(url, local_filename):
    # First, we get the response from the URL with the stream set to True to stream the download
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        # Retrieve the total file size by accessing the headers of the response
        total_size_in_bytes= int(r.headers.get('content-length', 0))
        # Initialize the progress bar
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(local_filename, 'wb') as f:
            # We loop over the content of the request in chunks
            for chunk in r.iter_content(chunk_size=1024):
                # We filter out keep-alive new chunks which are of size 0
                if chunk:
                    # Update the progress bar
                    progress_bar.update(len(chunk))
                    # Write the chunk to the file
                    f.write(chunk)
        # Close the progress bar
        progress_bar.close()
    return local_filename

# URL of the file to be downloaded
url = "http://www.interactome-atlas.org/data/HuRI.tsv"

# Local filename to save the downloaded file
local_filename = "data/HuRI.tsv"

# Call the download function
download_file(url, local_filename)
