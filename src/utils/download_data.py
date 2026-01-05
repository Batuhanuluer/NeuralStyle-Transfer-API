import os
import requests
import zipfile
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def download_coco_val(dest_folder="data/raw/coco"):
    """
    Downloads the COCO 2017 Validation set.
    """
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(dest_folder, "val2017.zip")
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        
    if not os.path.exists(zip_path):
        logger.info("Downloading COCO Val 2017... This might take a while.")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Download complete.")
        
        # Unzip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        logger.info("Unzipping complete.")
    else:
        logger.info("COCO dataset already exists.")

if __name__ == "__main__":
    download_coco_val()