import os 
import json
from utils import read_pascal_voc

filepath = '../../dataset/ocr_pad'
xml_dir = '../final_dataset_26_sep_23/train/labels'
all_files = [item.replace('.json','') for item in os.listdir(filepath)]

for filename in all_files:
    xml_name = f"{filename}.xml"
    xmlfile = os.path.join(xml_dir, xml_name)
    
    if os.path.exists(xmlfile):
        bboxes, labels = read_pascal_voc(xml_file=xmlfile)
        
        print(labels)
        break
        
        