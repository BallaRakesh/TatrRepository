import os 
import json
from tqdm import tqdm
import cv2

ocr_path = '/home/gayathri/table_processing/data/input/grassim/original_ocr'
dump_path = '/home/gayathri/table_processing/data/input/grassim/processed_ocr'

images_path = '/home/gayathri/table_processing/data/input/grassim/images'
# dump_viz = '/New_Volume/number_theory/table_processing/data/output/aabby_ocr/viz'

os.makedirs(dump_path, exist_ok=True)
ocr_files = os.listdir(ocr_path)


for file in tqdm(ocr_files, 'Processing'):
    # print(file)
    word_coordinates = {}
    img = cv2.imread(os.path.join(images_path, file.replace('json', 'png')))        
    img_h, img_w, _ = img.shape
    with open(os.path.join(ocr_path, file), 'r') as f:
        original_ocr = json.load(f)
        pages = original_ocr['layout']['pages']
        
        for page in pages:
            width = page['width']
            height = page['height']
            horizontal_resize = width / img_w
            vertical_resize = height / img_h
            
            texts = page['texts']
            tables = page['tables']
            # print(tables)
            for line in texts:
                
                for line in line['lines']:
                    for word_id, word in enumerate(line['words']):
                        if word_coordinates != {}:
                            word_id = max(word_coordinates.keys()) + 1
                        x1 = int(word['position']['l']//horizontal_resize)
                        y1 = int(word['position']['t']//vertical_resize)
                        x2 = int(word['position']['r']//horizontal_resize)
                        y2 = int(word['position']['b']//vertical_resize)
                        width = x2 - x1
                        height = y2 - y1
                        vertices = [(x1, y1), 
                                    (x1+width, y1), 
                                    (x1+width, y1+height), 
                                    (x1, y1+height)]
                        word_coordinates[word_id] = {
                                                    "text": word['text'],
                                                    "vertices": vertices,
                                                    "left": x1,
                                                    "top": y1,
                                                    "width": width,
                                                    "height": height,
                                                    "x1": x1,
                                                    "y1": y1,
                                                    "x2": x2,
                                                    "y2": y2,
                                                    'bbox': [x1, y1, x2, y2]
                                                }
                        # exit('LINES DONE')
            for table in tables:
                for cells in table['cells']:
                    # print(cells.keys())
                    for lines in cells['lines']:
                        for word_id, word in enumerate(lines['words']):
                            if word_coordinates != {}:
                                word_id = max(word_coordinates.keys()) + 1
                            # print(word, end='\n\n')
                            x1 = int(word['position']['l']//horizontal_resize)
                            y1 = int(word['position']['t']//vertical_resize)
                            x2 = int(word['position']['r']//horizontal_resize)
                            y2 = int(word['position']['b']//vertical_resize)
                            width = x2 - x1
                            height = y2 - y1
                            vertices = [(x1, y1), 
                                        (x1+width, y1), 
                                        (x1+width, y1+height), 
                                        (x1, y1+height)]
                            word_coordinates[word_id] = {
                                                        "text": word['text'],
                                                        "vertices": vertices
                                                    }
    with open(os.path.join(dump_path, file), 'w') as f:
        json.dump(word_coordinates, f, indent=4)
        
    
    # for word_idx in word_coordinates:
    #     x1 = word_coordinates[word_idx]['x1']
    #     y1 = word_coordinates[word_idx]['y1']
    #     x2 = word_coordinates[word_idx]['x2']
    #     y2 = word_coordinates[word_idx]['y2']
    #     # print(img.shape, word_idx, x1, y1, x2, y2)
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 244, 0), 2)
        
        
    # cv2.imwrite(os.path.join(dump_viz, file.replace('json', 'png')), img)



# for file in os.listdir(dump_path):
#     with open(os.path.join(dump_path, file), 'r') as f:
#         word_coordinates = json.load(f)
    # img = cv2.imread(os.path.join(images_path, file.replace('json', 'png')))        
    # for word_idx in word_coordinates:
    #     x1 = word_coordinates[word_idx]['x1']
    #     y1 = word_coordinates[word_idx]['y1']
    #     x2 = word_coordinates[word_idx]['x2']
    #     y2 = word_coordinates[word_idx]['y2']
    #     # print(img.shape, word_idx, x1, y1, x2, y2)
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 244, 0), 2)
        
        
    # cv2.imwrite(os.path.join(dump_viz, file.replace('json', 'png')), img)
    # exit()
