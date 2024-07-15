
from copy import copy
import os
import json
from functools import cmp_to_key
from tqdm import tqdm
from PIL import Image, ImageDraw
import shutil

def denormalize(h, w, bbox, denom=2):
    """
    Get entire label coordinate region
    
    Parameters
    ----------
    h: int
       heigth of the image
    w: int
       width of the image
    bbox: list
        word coordinates obtained from ocr
    denom: int
        Denominator for denormalize operation

    Returns:
    x0, y0, x1, y1: tuple
        tuple of denormalized word coordinates
    """

    x_center = float(bbox[0]) * w
    y_center = float(bbox[1]) * h
    width = float(bbox[2]) * w
    height = int(float(bbox[3]) * h)
    x0 = int(x_center - (width / denom))
    x1 = int(x_center + (width / denom))
    y0 = int(y_center - (height / denom))
    y1 = int(y_center + (height / denom))

    return x0, y0, x1, y1

def calculate_iou(bbox1, bbox2):
    """
    Finds the percentage of intersection  with a smaller box. (what percernt of smaller box is in larger box)
    """
    # assert bbox1['x1'] < bbox1['x2']
    # assert bbox1['y1'] < bbox1['y2']
    # assert bbox2['x1'] < bbox2['x2']
    # assert bbox2['y1'] < bbox2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    # min_area = min(bbox1_area,bbox2_area)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    intersection_percent = intersection_area / bbox2_area

    return intersection_percent

def contour_sort(a, b):
	if abs(a['y1'] - b['y1']) <= 15:
		return a['x1'] - b['x1']

	return a['y1'] - b['y1']

def get_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def get_text(ocr_region, labelled_region, words_coords=None, words=None, all_words=None):
    if get_area(labelled_region) > get_area(ocr_region):
        coords = []
        for idx, item in all_words.items():
            #if abs(item['bbox'][0] - labelled_region[0]) < 250:
                iou = calculate_iou(item['bbox'], labelled_region)
                if iou > 0.0:
                    coords.append({'bbox' : item['bbox'],
                                   'word': item['text']})
        # coords = sorted(coords, key=cmp_to_key(contour_sort))  
        # print(f"coords: {coords}")
        words = [item['word'] for item in coords]
        cords = [item['bbox'] for item in coords]
        assert len(words) == len(cords)
        return words, cords
    else:
        assert len(words_coords) == len(words)
        res = []
        coords = []
        for i in range(len(words_coords)):
            iou = calculate_iou(labelled_region, words_coords[i])
            #print(iou)
            if iou > 0.4:
                res.append(words[i])
                coords.append(words_coords[i])
        return res, coords


ocr_path = "custom_data/test/key_val_sets/"
all_words_path = "custom_data/test/all_words"

save_to = "custom_data/data_in_funsd_format/testing_data"
os.makedirs(os.path.join(save_to, 'images'), exist_ok=True)
os.makedirs(os.path.join(save_to, 'annotations'), exist_ok=True)

labels_path = '/home/gayathri/Downloads/CS/ANNOTATED_CS/ANNOTATED_VALIDATED_278/Labels'
images_path = '/home/gayathri/Downloads/CS/ANNOTATED_CS/ANNOTATED_VALIDATED_278/Images'

classes_path = "/home/gayathri/Downloads/CS/ANNOTATED_CS/ANNOTATED_VALIDATED_278/classes.txt"
with open(classes_path, 'r') as f:
    classes = f.readlines()

classes = [item.replace('\n','').strip() for item in classes]

images_list = os.listdir(images_path)
labels_list = os.listdir(labels_path)

labels_list = [item for item in labels_list if item.replace('txt', 'png') in images_list]

for labels in tqdm(labels_list, desc="Preparing"):
    with open(os.path.join(labels_path, labels), 'r') as f:
        label_data  = f.readlines()
    ocr_name = labels.replace('txt','json')
    with open(os.path.join(ocr_path, ocr_name) , 'r') as f:
        ocr_labels = json.load(f)
    with open(os.path.join(all_words_path, ocr_name), 'r')   as f:
        all_words = json.load(f)

    classes_enum = [int(item.split()[0]) for item in label_data]

    if max(classes_enum) > len(classes):
        continue

    
    
    label_data = [item.split()[1:] for item in label_data]
    label_data = [[float(item) for item in line ] for line in label_data]

    image_name = labels.replace('txt','png')
    image_loc = os.path.join(images_path, image_name)

    shutil.copy(image_loc, os.path.join(save_to,'images',image_name))

    image_org=Image.open(image_loc)
    image = Image.new('RGBA', image_org.size)
    image.paste(image_org)
    w, h = image_org.size

    denormalized_coords = [denormalize(h, w, coord) for coord in label_data]
    #print(denormalized_coords)

    draw=ImageDraw.Draw(image)

    
    #exit(s)
    for enum, coord in enumerate(denormalized_coords):
        draw.rectangle([coord[0], coord[1], coord[2], coord[3]], width=3 ,outline='blue') #, fill=(0, 0, 255, 125))
        draw.text((coord[0]+10, coord[1]-10), text=classes[classes_enum[enum]], fill='blue')
        
        
    #exit()
    ocr_labels_temp = copy(ocr_labels)
    labels_data_temp = copy(denormalized_coords)

    keep_coords = []

    id_counter = 0  

    # ToDo:
    # Two sepatate dicts for key data and value data 
    # to map the key-val relationships

    value_cntr = 1000
    key_cntr = 0

    value_dict = {}
    key_dict = {}

    covered_keys = []

    for i, ocr_coord in enumerate(ocr_labels):
        val_bbox = [int(item) for item in ocr_coord['value_bbox']]
        for j, label_coords in enumerate(labels_data_temp):
            if not len(val_bbox) == len(label_coords) == 0:
                # print(val_bbox, label_coords)
                iou = calculate_iou(val_bbox, label_coords)
                # print(iou, end='\n\n')
                if iou > 0:
                    overlapped_text, overlapped_coords = get_text(ocr_coord['value_bbox'],
                                               label_coords, 
                                               ocr_coord['value_text_bbox'], 
                                               ocr_coord['value_text'],
                                               all_words)
                    
                    # ocr_coord['key_text'] = [_ for item in ocr_coord['key_text'] if item in overlapped_text]
                    # if ocr_coord['key_text'] == []: 
                    #     key_text = None
                    # else: key_text = ocr_coord


                    """
                    # List to store all metadata
                    
                    keep_coords.append({
                        'id' : id_counter,
                        'key_bbox': ocr_coord['key_bbox'],
                        'key_text': ocr_coord['key_text'],
                        'key_text_bbox' : ocr_coord['key_text_bbox'],
                        'actual_key': classes[classes_enum[j]],
                        'actual_key_id': classes_enum[j],
                        'value_bbox': list(label_coords),
                        'value_text': overlapped_text,
                        'value_text_bbox': ocr_coord['value_text_bbox']
                    })

                    id_counter += 1
                    """
                    key_text = ''
                    #for kt in ocr_coord['key_text']:
                        #print(kt, ' :::: ', overlapped_text)
                    if ocr_coord['key_text'][0] in overlapped_text:
                        key_text = None
                        break

                    if key_text != None:
                        if key_cntr != 0 and (' '.join(ocr_coord['key_text']) == key_dict[key_cntr - 1]['text']):
                            
                            key_dict[key_cntr - 1]['linking'].append([ key_cntr-1,value_cntr])
                            key_cntr -= 1
                        else:
                            key_dict.update({key_cntr : { 
                                    'id' : key_cntr ,
                                    'box': ocr_coord['key_bbox'],
                                    'label': 'other',
                                    'text': ' '.join(ocr_coord['key_text']),
                                    'words' : [{'text': ocr_coord['key_text'][i], 
                                                'box':ocr_coord['key_text_bbox'][i] }
                                                for i in range(len(ocr_coord['key_text']))],
                                    'linking': [[key_cntr, value_cntr]]}})
                        
                            
                    #if overlapped_text != []:
                    value_dict.update(
                        {
                            value_cntr : {
                                'id' : value_cntr,
                                'box': list(label_coords),
                                'label': classes[classes_enum[j]],
                                'text': ' '.join(overlapped_text),
                                'words' : [{'text': overlapped_text[i], 
                                            'box':overlapped_coords[i] }
                                            for i in range(len(overlapped_text))],
                                'linking': [[key_cntr, value_cntr]]
                        }}
                    )
                    covered_keys.append(classes_enum[j])

                    # print(value_dict[value_cntr], end='\n\n')

                    key_cntr += 1
                    value_cntr += 1
                        #key_cntr += 1
                    

        draw.rectangle([int(ocr_coord['key_bbox'][0]), 
                        int(ocr_coord['key_bbox'][1]), 
                        int(ocr_coord['key_bbox'][2]), 
                        int(ocr_coord['key_bbox'][3])], 
                        width=3,
                        outline='red')
        draw.rectangle([int(ocr_coord['value_bbox'][0]), 
                        int(ocr_coord['value_bbox'][1]), 
                        int(ocr_coord['value_bbox'][2]), 
                        int(ocr_coord['value_bbox'][3])], 
                        width=3,
                        outline='green')
    all_actual_keys = [item for item in classes_enum]
    #all_ocr_keys = [item['actual_key_id'] for item in keep_coords]
    missed_keys = list(set(all_actual_keys) - set(covered_keys))
    if missed_keys != []:
        for i, key in enumerate(missed_keys):
            missed_key_idx = [classes_enum.index(item) for item in missed_keys]
            #print(missed_key_idx)
            value_bbox = [int(item) for item in denormalized_coords[missed_key_idx[i]]]
            value_text, value_coords =  get_text(ocr_region= [0, 0, 0, 0],
                                    labelled_region= value_bbox, 
                                    all_words= all_words)
              
            value_dict.update(
                        {
                            value_cntr : {
                                'id' : value_cntr,
                                'box': value_bbox,
                                'label': classes[key],
                                'text': ' '.join(value_text),
                                'words' : [{'text': value_text[i], 
                                            'box':value_coords[i] }
                                            for i in range(len(value_text))],
                                'linking': []
                        }}
                    )
            value_cntr += 1
             
            """
            keep_coords.append({
                    'id' : id_counter,
                   'key_bbox' : None,
                    'key_text' : None,
                    'key_text_bbox': None,
                    'actual_key': classes[key],
                    'actual_key_id': key,
                    'value_bbox': value_bbox,
                    'value_text':value_text,
                    'value_text_bbox': value_coords
              })
            
            id_counter += 1
            """

    #final_data = copy(key_dict)
    key_dict.update(value_dict)
    
    final_data = [key_dict[item] for item in key_dict]
    # for item in final_data:
    #     print(item, end="\n\n")
    # break

    with open(os.path.join(save_to,'annotations' ,ocr_name), 'w') as f:
        json.dump({"form" : final_data}, f, indent=4)