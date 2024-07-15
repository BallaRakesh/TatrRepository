import json
import os 
import cv2

data_path = '../padded_pred'
img_root_path = '../tables_cropped'
save_to = '../padded_pred_row'
os.makedirs(save_to, exist_ok=True)

all_data = os.listdir(data_path)
all_data = [item for item in all_data if item.endswith('_structure.json')]

for file in all_data:
    with open(os.path.join(data_path, file), 'r') as f:
        data = json.load(f)
        
    img_name = file.replace('_structure.json', '.png')
    img = cv2.imread(os.path.join(img_root_path, img_name)) 
    
    for table in data:
        if 'rows' in table:
            for row in table['rows']:
                bbox = row['bbox']
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 244, 0), 2)
        if 'columns' in table:
            for col in table['columns']:
                bbox = col['bbox']
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (244, 0, 0), 2)
    cv2.imwrite(os.path.join(save_to, img_name), img)
    
    
