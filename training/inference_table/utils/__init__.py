import os
import json
import numpy as np
from PIL import Image
from math import dist
import xml.etree.ElementTree as ET
import configparser
import pytesseract
import io
import copy
from google.cloud import vision_v1p4beta1 as vision
from row_detection import get_areas, bgs_in_images, binarize
import cv2

class ConfigParser:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


    # def __getattr__(self, name):
    #     return self.__dict__.get(name, None)

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
    if bbox2_area != 0:
        intersection_percent = intersection_area / bbox2_area

        return intersection_percent
    return 0

def get_mask_bbox(img):
    rows = np.any(img, axis=0)
    cols = np.any(img, axis=1)
    xmin, xmax = np.where(rows)[0][[0, -1]]
    ymin, ymax = np.where(cols)[0][[0, -1]]

    return [xmin, ymin, xmax, ymax]

def get_text_area(masked_image, main_bbox):
    # #print()
    # #print(masked_image.shape)
    
    roi = Image.fromarray(masked_image[main_bbox[1]:main_bbox[3], main_bbox[0]:main_bbox[2]], mode='RGB')
    roi_masked_image = Image.fromarray(np.zeros(masked_image.shape).astype('uint8'), mode='RGB')
    roi_masked_image.paste(roi, main_bbox)
    
    roi_masked_image = np.asarray(roi_masked_image)
    
    mask_bbox = get_mask_bbox(roi_masked_image)
    if mask_bbox[3]+mask_bbox[1] >= np.asarray(roi).shape[0]:
        mask_bbox[3] = np.asarray(roi).shape[0]
    return mask_bbox

def mask_all_text(image, ocr_words):
    image_blank = np.zeros(image.shape, np.uint16)
    pil_image = Image.fromarray(image_blank.astype('uint8'), mode='RGB')
    for word in ocr_words:
        word_bbox = [int(item) for item in ocr_words[word]['bbox']]
        
        h = word_bbox[3] - word_bbox[1]
        w = word_bbox[2] - word_bbox[0]
        
        text_mask = np.full((h, w, 3), 244)
    
        pil_text_mask = Image.fromarray(text_mask.astype('uint8'), mode='RGB')
        pil_image.paste(pil_text_mask, word_bbox)
        
    return np.asarray(pil_image)

def sort_coord(coords, return_order=False):
    points_dist = []
    for item in coords:
        
        points_dist.append(round(dist((0, 0), item[:2]), 2))
        # rearrange_pos.append(temp.index(min(temp)))
    sorted_order = np.argsort(points_dist)
    sorted_coords = [coords[i] for i in sorted_order]
    if not return_order:
        return sorted_coords
    else:
        return sorted_coords, sorted_order

def mask_header(masked_image, main_bbox, ocr_words):
    # #print()
    # #print(masked_image.shape)
    
    roi = Image.fromarray(masked_image[main_bbox[1]:main_bbox[3], main_bbox[0]:main_bbox[2]], mode='RGB')
    roi_masked_image = Image.fromarray(np.zeros(masked_image.shape).astype('uint8'), mode='RGB')
    roi_masked_image.paste(roi, main_bbox)
    
    roi_masked_image = np.asarray(roi_masked_image)
    
    mask_bbox = get_mask_bbox(roi_masked_image)
    if mask_bbox[3]+mask_bbox[1] >= np.asarray(roi).shape[0]:
        mask_bbox[3] = np.asarray(roi).shape[0]
    return mask_bbox


def get_text(area_bbox, ocr_words):
    text_metadata = []
    
    for word in ocr_words:
        # #print(area_bbox, ocr_words[word]['bbox'])
        # try:
        word_bbox = word['vertices'][0] + word['vertices'][2]
        
        iou = calculate_iou(area_bbox, word_bbox)
        # except Exception as e:
        #     #print(e)
            # iou = 0
        if iou > 0:
            # #print({'word':word['text'], 'bbox': word['bbox']})
            text_metadata.append({'text':word['text'], 'bbox': word_bbox, 'vertices': word['vertices']})
    # text = text.strip()
    return text_metadata


def remove_words(table_coords, table_ocr):

    new_table_ocr = copy.deepcopy(table_ocr)
    for word in table_ocr:
        word_y1 = table_ocr[word]['vertices'][0][1]
        word_y2 = table_ocr[word]['vertices'][2][1]
        if word_y1 < table_coords[1] and word_y2 < table_coords[1]:
            del new_table_ocr[word]
    del table_ocr
        
    return new_table_ocr

def read_pascal_voc(xml_file: str):

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

def get_cell(box1, box2):
    xmin = min(box1[0], box2[0])
    ymin = min(box1[1], box2[1])
    
    xmax = max(box1[2], box2[2])
    ymax = max(box1[3], box2[3])
    
    return [xmin, ymin, xmax, ymax]


def get_merged(bbox):
    xmin, ymin, xmax, ymax = [], [], [], []
    
    for box in bbox:
        xmin.append(box[0])
        ymin.append(box[1])
        xmax.append(box[2])
        ymax.append(box[3])
    
    try:    
        return [min(xmin), min(ymin), max(xmax), max(ymax)]
    
    except Exception:
        return []
    

def is_numerical(string:str):
    for ch in string:
        if ch.isalpha():
            return False
    return True

def adjust_table_coordinates(ocr_words ,table_coords):
    table_x1, table_y1, table_x2, table_y2 = table_coords
    new_table_bottom = table_y1


    if type(ocr_words) == dict:
        ocr_words = list(ocr_words.values())
    lines = get_lines(ocr_words, block_box=False)
    corner_items = {}
    
    # adjust_bottom_right = True
    corner_items = {}
    for line in lines:
        last_word = lines[line][len(lines[line])-1]
        # if table_x2 - (last_word['bbox'][2] + table_x1) <= ((table_x2 - table_x1) * 0.1):
        corner_items[line] = last_word

    for line in corner_items:
        if is_numerical(corner_items[line]['text']):
            # if corner_items[line]['bbox'][2] + table_x1 > new_table_right:
            #     new_table_right = corner_items[line]['bbox'][2] + table_x1
            if corner_items[line]['bbox'][3] + table_y1 > new_table_bottom:
                new_table_bottom = corner_items[line]['bbox'][3] + table_y1
        elif table_x2 - (corner_items[line]['bbox'][2] + table_x1) > ((table_x2 - table_x1) * 0.1):
            if corner_items[line]['bbox'][3] + table_y1 > new_table_bottom:
                new_table_bottom = corner_items[line]['bbox'][3] + table_y1

    
        # line_items = lines[line][len(lines[line]) - 1:len(lines[line])]
        # corner_items[line] = {'words':[], 'bbox':[]}
        # for word in line_items:
            
        #     if is_numerical(word['text']):
        #         if word['bbox'][2] + table_x1 > new_table_right:
        #             new_table_right = word['bbox'][2] + table_x1
        #         if word['bbox'][3] + table_y1 > new_table_bottom:
        #             new_table_bottom = word['bbox'][3] + table_y1

    # print('+++++++++++++++++++++++++++++++++')
    # print(new_table_bottom,)
    # print('+++++++++++++++++++++++++++++++++')
    return new_table_bottom
    
            
    
    

def box_intersects(bbox1, bbox2):
    
    
    iou = calculate_iou(bbox2,bbox1)
    if iou > 0:
        return True
    else:
        return False
    
    
def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def extract_text(ocr_client, image_path, save_to=None, filename=None):
    
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = ocr_client.text_detection(image=image)


    annotations = response.text_annotations
    
    word_coordinates = {}
    for i,text in enumerate(annotations):
        if i != 0:
            # #print('=' * 30)
            # #print(text.description)
            vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
            x1 = min([v.x for v in text.bounding_poly.vertices])
            x2 = max([v.x for v in text.bounding_poly.vertices])
            y1 = min([v.y for v in text.bounding_poly.vertices])
            y2 = max([v.y for v in text.bounding_poly.vertices])
            # #print('bounds: ' + str(vertices))
            if x2 - x1 == 0:
                x2 += 1
            if y2 - y1 == 0:
                y2 += 1
            
            ##print("Confidence: ", text.description, text.confidence)
            word_coordinates[i] = {
                "text": text.description,
                # "score" : text.confidence, 
                "vertices": vertices,
                # "left": x1,
                # "top": y1,
                # "width": x2 - x1,
                # "height": y2 - y1,
                # "x1": x1,
                # "y1": y1,
                # "x2": x2,
                # "y2": y2,
                # 'bbox': [x1, y1, x2, y2]
            }
            
    if save_to != None:
        os.makedirs(save_to, exist_ok=True)
        # filename = image_path.split('/')[-1].replace('png','json')
        with open(os.path.join(save_to, filename), 'w')  as f:
            json.dump(word_coordinates, f, indent=4)
    
    return word_coordinates

def sort_by_y(ocr_words):
    bboxes = [word['vertices'][0]+word['vertices'][2] for word in ocr_words]
    sort_index = [i for i, x in sorted(enumerate(bboxes), key=lambda x: x[1][1])]
    return [ocr_words[i] for i in sort_index]
    
    

def get_lines(image, coords=None, block_box=True):
    
    # # # image_copy = copy.deepcopy(image)
    # # # gray_image = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
    # # # threshold, bin_image = binarize(gray_image)
    # # # main_bg, header_bg, white_pixels, black_pixels = bgs_in_images(bin_image, threshold)
    # # # boxes, _ = get_areas(bin_image, main_bg=main_bg, header_bg=header_bg)
    # # # boxes = boxes['row_bboxes']

    # # # lines = {}
    # # # for i in range(len(boxes)-1) :
    # # #     if not boxes[i][1] == boxes[i][3] :
    # # #         lines[i] = {'block_bbox':boxes[i], 'words':[]}

    # # #         if not coords == None:
    # # #             block_words = get_text(area_bbox=boxes[i], ocr_words=coords)
    # # #             lines[i]['words'] = block_words
    # # # return lines


    font_size = get_font_size(coords)
    line_num = 0
    lines = {}
    
    coords = sort_by_y(coords)
    
    for i in range(len(coords)-1):
        word_dist = coords[i+1]['bbox'][1] - coords[i]['bbox'][1]
        # #print(word_dist, font_size)
        if word_dist <= (font_size) + 1:
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
    if len(lines) != 0:
        lines[len(lines)-1].append(coords[i+1])
    
    if block_box:
        for item in lines:

            line_word_bboxes = [word['bbox'] for word in lines[item]]
            _, sort_order = sort_coord(line_word_bboxes, return_order=True)
            x_min = min([x['bbox'][0] for x in lines[item]])
            x_max = max([x['bbox'][2] for x in lines[item]])
            
            y_min = min([y['bbox'][1] for y in lines[item]])
            y_max = max([y['bbox'][3] for y in lines[item]])
            lines[item] = {
                'block_bbox': [x_min, y_min, x_max, y_max],
                'words': [lines[item][i] for i in sort_order]
            }   
        return lines
    
    else:
        for line in lines:
            bboxes = [word['bbox'] for word in lines[line]]
            _, sort_order = sort_coord(bboxes, return_order=True)

            new_order = [lines[line][i] for i in sort_order]
            lines[line] = new_order
        return lines




def get_font_size(coords):
    
    for coord in coords:
        
        coord.update({'bbox': coord['vertices'][0]+coord['vertices'][2]})
    # coords = [item['bbox'] for item in coords] 
    # coords = sort_coord(coords)
    try:
        font_size = round(np.mean([coord['bbox'][3]-coord['bbox'][1] for coord in coords ]))
    except:
        font_size = 1
    return font_size


def get_ocr_client(ocr_name, credentials=None):
    if ocr_name == 'gv':
        return vision.ImageAnnotatorClient.from_service_account_json(credentials)
    elif ocr_name == 'tesseract':
        return None
    elif ocr_name == 'abby':
        return None
    elif ocr_name == 'kadmos':
        return None
    return None


def perform_ocr(ocr, ocr_client=None, image_content=None, doc_ocr=None, table_coords=None):
    if ocr == 'gv':
        return ExtractText.google_vision(ocr_client, image_content)
    elif ocr == 'tesseract':
        return ExtractText.tesseract(image_content)
    elif ocr == 'abby':
        return None
    elif ocr == 'kadmos':
        return None
    elif ocr == 'other':
        return ExtractText.custom_ocr(doc_ocr=doc_ocr, table_coords=table_coords)
    

def clean_ocr(ocr_data):
    final_ocr = {}
    bboxes = [ocr_data[word]['vertices'][0]+ocr_data[word]['vertices'][2] for word in ocr_data] 
    words = [ocr_data[word]['text'] for word in ocr_data]
    
    ocr_words = zip(bboxes, words)
    sorted_words = sorted(ocr_words, key=lambda x: x[1])

    for i in range(len(sorted_words)):
        final_ocr[i] = {
            'text' : sorted_words[i][1],
            'vertices': [
                [bboxes[i][0][0], bboxes[i][0][1]],
                [bboxes[i][0][2], bboxes[i][0][1]],
                [bboxes[i][0][2], bboxes[i][0][3]],
                [bboxes[i][0][0], bboxes[i][3]]
            ]
        }
    # bboxes, sort_order = sort_coord(bboxes, return_order=True)
    
    # words_reordered = [words[i] for i in sort_order]
    
    
    # for i in range(len(words_reordered)):
    #     final_ocr[i] = {
    #         'text' : words_reordered[i],
    #         'vertices': [
    #             [bboxes[i][0], bboxes[i][1]],
    #             [bboxes[i][2], bboxes[i][1]],
    #             [bboxes[i][2], bboxes[i][3]],
    #             [bboxes[i][0], bboxes[i][3]]
    #         ]
    #     }
    
    return final_ocr
class ExtractText:
    def __init__(self):
        pass
    
    def tesseract(image_content, language='eng'):



        # Open the image
        img = Image.open(io.BytesIO(image_content))

        # Perform OCR on the image
        data = pytesseract.image_to_data(img, lang=language, output_type='dict')

        # Extract word bounding boxes
        word_coordinates = {}
        cnt = 0
        for i, text in enumerate(data.get('text', [])):
            if text.strip():  # Skip empty strings
                x1, y1, width, height = map(int, (data['left'][i], data['top'][i], data['width'][i], data['height'][i]))
                x2, y2 = x1 + width, y1 + height

                # vertices = [
                #     {"x": x1, "y": y1},
                #     {"x": x2, "y": y1},
                #     {"x": x2, "y": y2},
                #     {"x": x1, "y": y2}
                # ]

                vertices = [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ]
                cnt+=1
                word_id = cnt
                word_coordinates[word_id] = {
                    "text": text,
                    "vertices": vertices,
                    # "left": x1,
                    # "top": y1,
                    # "width": width,
                    # "height": height,
                    # "x1": x1,
                    # "y1": y1,
                    # "x2": x2,
                    # "y2": y2,
                    # 'bbox': [x1, y1, x2, y2]
                }

        return word_coordinates
    
    def google_vision(ocr_client, image_content):
    
        image = vision.Image(content=image_content)
        response = ocr_client.text_detection(image=image)


        annotations = response.text_annotations
        
        word_coordinates = {}
        for i,text in enumerate(annotations):
            if i != 0:
                # #print('=' * 30)
                # #print(text.description)
                vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
                x1 = min([v.x for v in text.bounding_poly.vertices])
                x2 = max([v.x for v in text.bounding_poly.vertices])
                y1 = min([v.y for v in text.bounding_poly.vertices])
                y2 = max([v.y for v in text.bounding_poly.vertices])
                # #print('bounds: ' + str(vertices))
                if x2 - x1 == 0:
                    x2 += 1
                if y2 - y1 == 0:
                    y2 += 1
                
                ##print("Confidence: ", text.description, text.confidence)
                word_coordinates[i] = {
                    "text": text.description,
                    # "score" : text.confidence, 
                    "vertices": vertices,
                    # "left": x1,
                    # "top": y1,
                    # "width": x2 - x1,
                    # "height": y2 - y1,
                    # "x1": x1,
                    # "y1": y1,
                    # "x2": x2,
                    # "y2": y2,
                    # 'bbox': [x1, y1, x2, y2]
                }
                
            
        return word_coordinates

    def custom_ocr(doc_ocr, table_coords):
        
        table_x1, table_y1, table_x2, table_y2  = table_coords
            
        table_ocr = {}
        for word in doc_ocr:
            word_x1, word_y1 = doc_ocr[word]['vertices'][0]
            word_x2, word_y2 = doc_ocr[word]['vertices'][2]

            if not word_x2 - word_x1 <= 2 and not word_y2 - word_y1 <= 2 and not doc_ocr[word]['text'] == '|':
                if table_x1 > word_x1:  table_x1 = word_x1
                if table_x2 < word_x2:  table_x2 = word_x2
                
                if table_y1 > word_y1 and table_y1 < word_y2: table_y1 = word_y1

                if table_y2 > word_y1 and table_y2 < word_y2: table_y2 = word_y1
                
                if table_x1-1 <= word_x1 <= word_x2 <= table_x2+1 and \
                    table_y1-1 <= word_y1 <= word_y2 <= table_y2+1:
                        
                        new_word_x1 = word_x1 - table_x1
                        new_word_y1 = word_y1 - table_y1
                        new_word_x2 = word_x2 - table_x1
                        new_word_y2 = word_y2 - table_y1
                        
                        table_ocr[word] = {
                            'text' : doc_ocr[word]['text'],
                            'vertices' : [
                                            [new_word_x1, new_word_y1],
                                            [new_word_x2, new_word_y1],
                                            [new_word_x2, new_word_y2],
                                            [new_word_x1, new_word_y2]
                                        ]
                            }
        #             #print(table_ocr[word])
        # #print('%'*20)
        table_coords = [table_x1, table_y1, table_x2, table_y2]
        return table_ocr, table_coords
        