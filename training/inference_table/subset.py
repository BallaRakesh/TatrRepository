import os 
from tqdm import tqdm
import shutil

table_images_path = "/New_Volume/master_table_extraction/data/input/ingram_test_subset"
doc_images_path = "/New_Volume/master_table_extraction/data/input/ingram"

save_to =  "/New_Volume/master_table_extraction/data/input/ingram_test"
os.makedirs(save_to, exist_ok=True)

table_images = os.listdir(table_images_path)

for item in tqdm(table_images, 'Processing'):
    doc_name = f"{item.split('_')[0]}.png"
    
    shutil.copy(os.path.join(doc_images_path, doc_name), 
                os.path.join(save_to, doc_name))