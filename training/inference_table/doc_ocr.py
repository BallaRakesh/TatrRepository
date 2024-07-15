import os
import json
from utils import perform_ocr, get_ocr_client
from tqdm import tqdm

ocr = 'gv'
credentials_path = "./utils/gvkey.json"

ocr_client = get_ocr_client(ocr, credentials_path)


save_to = '/home/gayathri/table_processing/data/input/table_images/ocr'
os.makedirs(save_to, exist_ok=True)

doc_path = '/home/gayathri/table_processing/data/input/table_images/images'
docs_list = os.listdir(doc_path)


for doc in tqdm(docs_list, 'Reading words'):

    with open(os.path.join(doc_path, doc), 'rb') as image_file:
            content = image_file.read()

    words = perform_ocr(ocr, ocr_client, image_content = content)
    jsonname = doc.replace('png', 'json')
    with open(os.path.join(save_to, jsonname), 'w') as f:
        json.dump(words, f, indent=4)