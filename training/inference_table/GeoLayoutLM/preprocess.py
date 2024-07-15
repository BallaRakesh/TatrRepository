import os 
import json
from google.cloud import vision
from base64 import b64encode
import argparse
from transformers import BertTokenizer
import numpy as np
from math import dist
import cv2
from PIL import Image
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
# import sys
# sys.append('../')
import utils


# def get_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--images_dir',
#                         help="Directory where the images to process are")
#     parser.add_argument('--ann_dir',
#                         help="Directory where the annotations to process are")
#     parser.add_argument('--ocr_dir',
#                         help="Directory where the OCR content are")
#     parser.add_argument('--dump_dir',
#                         help="Path to where the generated data must be dumped to")
#     parser.add_argument('--dataset_type',
#                         choices=['testing_data', 'training_data'],
#                         help='Select the type of dataset that needs to be generated')
    
#     return parser.parse_args()

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

def read_xml(xml_file):
    # #print(f"XML NAME: {xml_file}")
    if not os.path.exists(xml_file):
        xml_root_path = '/'.join(xml_file.split('/')[:-1])
        xml_file_name = xml_file.split('/')[-1]
        xml_file_name = xml_file_name.split('_')[0]
        xml_file = f"{xml_root_path}/{xml_file_name}.xml"
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    labels = []

    for object_ in root.iter('object'):

        ymin, xmin, ymax, xmax = None, None, None, None
        
        label = object_.find("name").text

        for box in object_.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)

        bbox = [xmin, ymin, xmax, ymax] # PASCAL VOC
        
        bboxes.append(bbox)
        labels.append(label)

    return bboxes, labels

# OCR Vision function
# def get_ocr_vision_api(ocrpath, jsonname, file):
#     image = file
#     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "digital-hall-399509-cae6bb37c802.json"
#     ctxt = b64encode(image.read()).decode()
#     client = vision.ImageAnnotatorClient()
#     image = vision.Image(content=ctxt)

#     response = client.text_detection(image=image)

#     # for res in response.text_annotations:
#     # 	#print(res.confidence)

#     word_coordinates = []
#     for i, text in enumerate(response.text_annotations):
#         if i != 0:
#             vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
#             x1 = min([v.x for v in text.bounding_poly.vertices])
#             x2 = max([v.x for v in text.bounding_poly.vertices])
#             y1 = min([v.y for v in text.bounding_poly.vertices])
#             y2 = max([v.y for v in text.bounding_poly.vertices])
#             if x2 - x1 == 0:
#                 x2 += 1
#             if y2 - y1 == 0:
#                 y2 += 1
#             """"left": x1,
#                 "top": y1,
#                 "width": x2 - x1,
#                 "height": y2 - y1,"""
#             word_coordinates.append({
#                 "text": text.description,
#                 "left": x1,
#                 "top": y1,
#                 "width": x2 - x1,
#                 "height": y2 - y1,
#                 "x1": x1,
#                 "y1": y1,
#                 "x2": x2,
#                 "y2": y2
#             })
#         else:
#             all_text = text.description
#     save_coords = {}
#     for i in range(len(word_coordinates)):
#         save_coords[i] = {
#             "text": word_coordinates[i]['word'],
#             "score": 0.0,
#             "left": word_coordinates[i]['left'],
#             "top": word_coordinates[i]['top'],
#             "width": word_coordinates[i]['width'],
#             "height": word_coordinates[i]['height'],
#             "x1": word_coordinates[i]['x1'],
#             "y1": word_coordinates[i]['y1'],
#             "x2": word_coordinates[i]['x2'],
#             "y2": word_coordinates[i]['y2'],
#             "bbox": [word_coordinates[i]['x1'], word_coordinates[i]['y1'], 
#                     word_coordinates[i]['x2'], word_coordinates[i]['y2']]
#         }
#     if ocrpath is not None:
#         with open(os.path.join(ocrpath, jsonname), 'w') as f:
#             json.dump(save_coords, f, indent=4)
#     else:
#         with open(f"/New_Volume/number_theory/GeoLayoutLM/datahub/ocr/{jsonname}", 'w') as f:
#             json.dump(save_coords, f, indent=4)
#     # im_name = list(file.split('/'))[-1]
#     return word_coordinates


def read_ocr(file):
    # #print("called Image OCR...")
    word_coordinates = []

    ocr_path = os.path.join(file)
    # #print(ocr_path)
    with open(ocr_path, 'r') as f:
        ocr_data = json.load(f)
    for data in ocr_data:
        word_coordinates.append({
            'word': ocr_data[data]['text'],
            'x1': ocr_data[data]['x1'],
            'y1': ocr_data[data]['y1'],
            'x2': ocr_data[data]['x2'],
            'y2': ocr_data[data]['y2']
        })

    return word_coordinates

def sort_coords(coords, axis=None):
    if axis == 1:
        points = [[item['bbox'][1], item['bbox'][3]] for item in coords]
    elif axis == 0:
        points = [[item['bbox'][0], item['bbox'][2]] for item in coords]
    else:
        points = [item['bbox'][:2] for item in coords]
    points_dist = []
    for coord in points:
        points_dist.append(round(dist((0, 0), coord), 2))
    sorted_order = np.argsort(points_dist)
    sorted_coords = [coords[i] for i in sorted_order]
    return sorted_coords

def get_lines(font_size, coords):
    line_num = 0
    lines = {}
    for i in range(len(coords)-1):
        word_dist = coords[i+1]['bbox'][1] - coords[i]['bbox'][1]
        if word_dist <= font_size+1: ##### Update
            if not line_num in lines:
                lines.update({line_num: [coords[i]]})
            else:
                lines[line_num].append(coords[i])
        else:
            if not line_num in lines:
                lines.update({line_num: [coords[i]]})
            else:
                lines[line_num].append(coords[i])
            line_num += 1
            # lines.update({line_num: [coords[i+1]]})
    lines[len(lines)-1].append(coords[i+1])
    for item in lines:
        x_min = min([x['bbox'][0] for x in lines[item]])
        x_max = max([x['bbox'][2] for x in lines[item]])
        
        y_min = min([y['bbox'][1] for y in lines[item]])
        y_max = max([y['bbox'][3] for y in lines[item]])
        lines[item] = {
            'block_bbox': [x_min, y_min, x_max, y_max],
            'words': lines[item]
        }
        
    return lines

def get_line_num(lines, start_num):
    for line in range(start_num, -1, -1):
        if lines[line] != None:
            return line

def get_seperators(lines, min_word_threshold):
    for line in lines:
        if abs(lines[line]['block_bbox'][2]-lines[line]['block_bbox'][0]) <= min_word_threshold:
            if line == 0:
                continue
            merge_with_line = get_line_num(lines, line-1)  
            # #print(line,'merge with', merge_with_line)      
            lines[merge_with_line]['words'].append(lines[line]['words'])
            lines[line] = None
            
    blocks = {}
    for line in lines:
        if lines[line] != None:
            blocks[line] = lines[line]
    return blocks

def correct_header_row(coords, labels):
    if 'table column header' in labels:
        header_idx = labels.index('table column header')
        labels[header_idx] = ''
        for i in range(len(coords)):
            if labels[i] == 'table row':
                iou = calculate_iou(coords[header_idx], coords[i])
                if iou >= 0.9:
                    labels[i] = 'table column header'
        del labels[header_idx]
        del coords[header_idx]
    return coords, labels

def get_labels(ann_path, blocks):
    gt_coords, gt_labels = read_xml(ann_path)
    gt_coords, gt_labels = correct_header_row(gt_coords, gt_labels)
    keep_coords = ['table column header', 'table row', 'trash']
    for block in blocks:
        for i in range(len(gt_coords)):
            if gt_labels[i] in keep_coords:
                block_coords = blocks[block]['block_bbox']
                row_coord = gt_coords[i]
                iou = calculate_iou(row_coord, block_coords)
                if iou > 0:
                    blocks[block].update({"block_label": gt_labels[i]})
    
    return blocks
            
            
# def get_blocks(coords):
#     """
#     Each row (words with same y-min) will become a block
#     """
#     for coord in coords:
#         coord.update({'bbox': [coord['x1'], coord['y1'], coord['x2'], coord['y2']]})
#     # coords = [[item['x1'], item['y1'], item['x2'], item['y2']] for item in coords] 
#     coords = sort_coords(coords, axis=1)
    
#     font_size = round(np.mean([coord['y2']-coord['y1'] for coord in coords ]))
    
#     lines = get_lines(font_size, coords)
    
    
#     line_distances = []
#     for ln in lines:
#         if not ln == len(lines)-1:
#             line_distances.append(lines[ln+1]['block_bbox'][1] - lines[ln]['block_bbox'][3])
            
#     # #print(line_distances)
    
#     return lines
                

def get_data(ocr_data, image, blocks):
    # args = get_args()

    # ann_path = args.ann_dir
    # images_path = args.images_dir
    # ocr_path = args.ocr_dir
    # dump_path = args.dump_dir
    # dataset_type = args.dataset_type
    
    # if os.path.exists(dump_path):
    #     shutil.rmtree(dump_path)
    
    # if os.path.exists(os.path.join(dump_path, 'preprocessed_files_val.txt')):
    #     shutil.rmtree(os.path.join(dump_path, 'preprocessed_files_val.txt'))
    # if not os.path.exists(dump_path):
    #     os.makedirs(dump_path)
    #     # os.makedirs(os.path.join(dump_path, 'training_data'))
    #     # os.makedirs(os.path.join(dump_path, 'training_data', 'images'))
    #     # os.makedirs(os.path.join(dump_path, 'training_data', 'annotations'))
    # if not os.path.exists(f"{dump_path}/{dataset_type}"):
    #     os.makedirs(os.path.join(dump_path, dataset_type))
    #     os.makedirs(os.path.join(dump_path, dataset_type, 'images'))
    #     os.makedirs(os.path.join(dump_path, dataset_type, 'annotations'))
    
    VOCA = "bert-base-uncased"
    
    tokenizer = BertTokenizer.from_pretrained(VOCA, do_lower_case=True)
    
    # images_list = os.listdir(images_path)
    # ocr_list = os.listdir(ocr_path)
    CLASSES = ['data_cell', 'header_cell', 'trash', 'O']
    # with open(os.path.join(dump_path, 'class_names.txt'), 'w') as f:
    #     for item in CLASSES:
    #         f.writelines(f"{item}\n")
        
    # for imgname in tqdm(images_list, 'Processing'):
        # #print(imgname)
        # jsonname = imgname.replace('png','json')
        # if ocr_path != None and jsonname in ocr_list:
        #     coords = read_ocr(os.path.join(ocr_path, jsonname))
        # else:
        #     img = open(os.path.join(images_path, imgname), 'rb')
        #     coords = get_ocr_vision_api(ocr_path, jsonname, img)
        #     img.close()
        
    # try:
    
    # blocks = utils.get_lines(ocr_data)
    # except:
    #     blocks = None
    #     pass
    
    # if blocks != None:
        # if ann_path != None:
        #     ann_name = imgname.replace('png', 'xml')
        #     xml_path = os.path.join(ann_path, ann_name) 
        #     blocks = get_labels(xml_path, blocks)
            
            
        
        # img = cv2.imread(os.path.join(images_path, imgname))
        # for item in blocks:
        #     coords = blocks[item]['block_bbox']
        #     cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 244, 0), 2)
        # cv2.imwrite(os.path.join('block_viz', imgname), img)
            
    prepare_data_struct = {
        "meta": {
        "image_path": str,
        "imageSize": {
            "width": int,
            "height": int
            },
            "voca": str
        },
        "blocks": {
            "first_token_idx_list": [],
            "boxes": [],
        },
        "words": [],
        'parse': {
            'class': {},
            'relations': []
        }
    }
    
    for c in CLASSES:
        prepare_data_struct["parse"]['class'].update({c: []})
    
    # image_file = os.path.join(images_path, imgname)
    image_h, image_w = image.shape[:2]
    
    # prepare_data_struct['meta']['image_path'] = image_file
    prepare_data_struct['meta']['imageSize']['width'] = image_w
    prepare_data_struct['meta']['imageSize']['height'] = image_h
    prepare_data_struct['meta']['voca'] = "bert-base-uncased"

    num_tokens = 0
    
    
    for block in blocks:
        real_word_idx = 0
        prepare_data_struct['blocks']['boxes'].append(blocks[block]['block_bbox'])
        
        class_seq = []
        for coord in blocks[block]['words']:
            word_text = coord["text"]
            
                
            # bb = [coord['vertices']["x1"], coord['vertices']["y1"], coord['vertices']["x2"], coord['vertices']["y2"]]
            bb = coord['vertices'] #[[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_text))
        
            word_obj = {"text": word_text, "tokens": tokens, "boundingBox": bb}
            prepare_data_struct['words'].append(word_obj)
                
            if real_word_idx == 0:
                prepare_data_struct['blocks']['first_token_idx_list'].append(num_tokens+1)
            num_tokens+=len(tokens)
            
            class_seq.append(len(prepare_data_struct["words"]) - 1) # word index
            
            real_word_idx += 1
            
        if 'block_label' in blocks[block]:
            # #print(blocks[block]['block_label'])
            if blocks[block]['block_label'] == 'table row':
                label = 'data_cell'
            elif blocks[block]['block_label'] == 'table column header':
                # #print("HEADER CELL")
                label = 'header_cell'
            elif blocks[block]['block_label'] == 'trash':
                label = 'trash'
            else:
                label = 'O'
                
            prepare_data_struct["parse"]['class'][label].append(class_seq)
        
        # shutil.copy(image_file,
        #             os.path.join(dump_path, dataset_type, 'images', imgname))
        
        # with open(os.path.join(dump_path, dataset_type, 'annotations', jsonname), 'w') as f:
        #     json.dump(prepare_data_struct, f, indent=4)
        
        # preprocess_type = 'train'
        # if dataset_type == 'testing_data':
        #     preprocess_type = 'val'
        # with open(os.path.join(dump_path, f'preprocessed_files_{preprocess_type}.txt'), 'a+') as f:
        #     #print(os.path.join(dataset_type, 'annotations', jsonname), file=f)
    return prepare_data_struct
# if __name__ == '__main__':
#     main()