import os
import json
import pandas as pd
import csv

segragation_path = 'segragation.json'
with open(segragation_path, 'r') as f:
    seg_info = json.load(f)
    

root_path = '/New_Volume/number_theory/table-transformer-main/final_dataset_26_sep_23/columns'
# all_img_res = {}
res_csv_path = os.path.join(root_path, 'overall.csv')
with open(res_csv_path, 'r', encoding = 'utf-8') as f:
    csv_res = csv.DictReader(f)
    
    image_res = {}
    for item in csv_res:
        
        if 'Image name' in item:
            image_res[item['Image name']] = item
            
images_list = image_res.keys()
print(images_list)
print(len(images_list))
final_res = []

for cat in seg_info:
    all_horizontal = []  
    all_lines  = []
    all_verticals = []  
    no_lines = []
    skewed = []
    for item in seg_info[cat]:
        for im in seg_info[cat][item]:
            if im in images_list:
                
                if item == 'all_horizontal':
                    all_horizontal.append(image_res[im])
                elif item == 'all_lines':
                    all_lines.append(image_res[im])
                elif item == 'all_verticals':
                    all_verticals.append(image_res[im])
                elif item == 'skewed':
                    skewed.append(image_res[im])
                elif item == 'no_lines':
                    no_lines.append(image_res[im])
     
    save_res = os.path.join(root_path, 'category_wise', cat)
    print(save_res)
    os.makedirs(save_res, exist_ok=True)

    pd.DataFrame(all_horizontal).to_csv(os.path.join(save_res, 'all_horizontal.csv'), index=False)
    pd.DataFrame(all_verticals).to_csv(os.path.join(save_res, 'all_verticals.csv'), index=False)
    pd.DataFrame(all_lines).to_csv(os.path.join(save_res, 'all_lines.csv'), index=False)
    pd.DataFrame(no_lines).to_csv(os.path.join(save_res, 'no_lines.csv'), index=False)
    pd.DataFrame(skewed).to_csv(os.path.join(save_res, 'skewed.csv'), index=False)

# with open(os.path.join(save_res, 'all_horizontal.csv'), 'w') as f:
#     json.dump(all_horizontal, f, indent=4)

# with open(os.path.join(save_res, 'all_verticals.csv'), 'w') as f:
#     json.dump(all_verticals, f, indent=4)
    
# with open(os.path.join(save_res, 'all_lines.csv'), 'w') as f:
#     json.dump(all_lines, f, indent=4)
    
# with open(os.path.join(save_res, 'no_lines.csv'), 'w') as f:
#     json.dump(no_lines, f, indent=4)
    
# with open(os.path.join(save_res, 'skewed.csv'), 'w') as f:
#     json.dump(skewed, f, indent=4)