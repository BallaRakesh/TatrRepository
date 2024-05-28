"""
python3 align_boxes.py --images_path ../benchmark/cropped_pad/images --pred_path ../padded_pred --out_path ../padded_pred/xmls --ocr_path ../ocr

Post processing for TaTR model

Author: Gayathri Satheesh
Created on: Sep 21, 2023
Updated on:

"""

import os
import json

import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
from math import dist

from copy import copy, deepcopy

import xml.etree.ElementTree as ET

from gen_xml import generate_xml

from skimage.measure import label, regionprops

import cv2

from PIL import Image

import matplotlib.pyplot as plt

from collections import defaultdict

import matplotlib.patches as patches
from matplotlib.patches import Patch

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

def read_pascal_voc(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    labels = []

    for object_ in root.iter('object'):

        ymin, xmin, ymax, xmax = None, None, None, None
        
        label = object_.find("name").text

        for box in object_.findall("bndbox"):
            ymin = int(float(box.find("ymin").text))
            xmin = int(float(box.find("xmin").text))
            ymax = int(float(box.find("ymax").text))
            xmax = int(float(box.find("xmax").text))

        bbox = [xmin, ymin, xmax, ymax] # PASCAL VOC
        
        bboxes.append(bbox)
        labels.append(label)

    return bboxes, labels

def read_json(pred_file: str):
    with open(pred_file, 'r') as f:
        data = json.load(f)
    if len(data) > 1: # since tables are cropped only one table should be present in each image
        return 0, 0
    if not data == []:
        data = data[0]
        
        bboxes = []
        labels = []
        
        col_headers = 0
        
        for key in data:
            for pred in data[key]:
                try: 
                    if "column header" in pred:
                        if pred["column header"]:
                            col_headers += 1
                    bboxes.append([int(item) for item in pred['bbox']])
                    labels.append(pred['label'])
                except Exception as e: 
                    # print(e)
                    pass
            
        return bboxes, labels, col_headers
    return [], [], []

def get_mask_bbox(img):
    rows = np.any(img, axis=0)
    cols = np.any(img, axis=1)
    xmin, xmax = np.where(rows)[0][[0, -1]]
    ymin, ymax = np.where(cols)[0][[0, -1]]

    return [xmin, ymin, xmax, ymax]

def get_text_area(masked_image, main_bbox, ocr_words):
    # print()
    # print(masked_image.shape)
    
    roi = Image.fromarray(masked_image[main_bbox[1]:main_bbox[3], main_bbox[0]:main_bbox[2]], mode='RGB')
    roi_masked_image = Image.fromarray(np.zeros(masked_image.shape).astype('uint8'), mode='RGB')
    roi_masked_image.paste(roi, main_bbox)
    
    roi_masked_image = np.asarray(roi_masked_image)
    
    mask_bbox = get_mask_bbox(roi_masked_image)
    if mask_bbox[3]+mask_bbox[1] >= np.asarray(roi).shape[0]:
        mask_bbox[3] = np.asarray(roi).shape[0]
    return mask_bbox
    # for word in ocr_words:
    #     word_bbox = [int(item) for item in ocr_words[word]['bbox']]
    #     if calculate_iou(word_bbox, main_bbox) > 0:
    #         text_region.append(ocr_words[word]['bbox'])
    # if len(text_region) > 0:
    #     # print(text_region)
    #     sorted_text_region = sort_coord(text_region)
    #     # print(sorted_text_region)
    #     x_mins = []
    #     y_mins = []
    #     x_maxs = []
    #     y_maxs = []
    #     for item in sorted_text_region:
    #         x_mins.append(item[0])
    #         y_mins.append(item[1])
    #         x_maxs.append(item[2])
    #         y_maxs.append(item[3])

    #     text_region_boundary = [min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)]
    #     return text_region_boundary
    # return []

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

def align_y(coords):
    points = [item[1] for item in coords]
    alignment = []
    # points_dist = []
    # for item in coords:
        
    #     points_dist.append(round(dist((0, 0), item[:2]), 2))
    #     # rearrange_pos.append(temp.index(min(temp)))
    sorted_order = np.argsort(points)
    sorted_coords = [coords[i] for i in sorted_order]
    
    
    return sorted_coords

def sort_coord(coords):
    points_dist = []
    for item in coords:
        
        points_dist.append(round(dist((0, 0), item[:2]), 2))
        # rearrange_pos.append(temp.index(min(temp)))
    sorted_order = np.argsort(points_dist)
    sorted_coords = [coords[i] for i in sorted_order]
    return sorted_coords

    # if sort_key == 'row':
    #     pos = 1
    # else:
    #     pos = 0
       
    # poitional_vals = [item[pos] for item in coords]
    # sorted_order = np.argsort(poitional_vals)
    # sorted_coords = [coords[i] for i in sorted_order]
    
    # return sorted_coords

def rearrange_coords(gt_coords, pred_coords):
    rearrange_pos = []
    for gt_item in gt_coords:
        temp = []
        for pred_item in pred_coords:
            points_dist = round(dist(gt_item[:2], pred_item[:2]), 2)
            temp.append(points_dist)
        rearrange_pos.append(temp.index(min(temp)))
    rearranged_coords = [pred_coords[i] for i in rearrange_pos]
    return rearranged_coords

def align_coords(coords, axis='row'):
    if axis == 'row':
        pos1 = 0
        pos2 = 2
        
    else:
        pos1 = 1
        pos2 = 3
        
    min_coords = []
    max_coords = []
    
    for coord in coords:
        min_coords.append(coord[pos1])
        max_coords.append(coord[pos2])
    
    min_coord = min(min_coords)
    max_coord = max(max_coords)
    
    for coord in coords:
        coord[pos1] = min_coord
        coord[pos2] = max_coord

    
    for i in range(len(coords)):
        if i != len(coords)- 1:
            if axis == 'row':
                coords[i][3] = coords[i+1][1]
            else:
                coords[i][2] = coords[i+1][0]
            # coords[i][2] = coords[i-1][end_pos1]
            # coords[i][3] = coords[i-1][end_pos2]
    # print(coords)
    # exit()
    # print(axis)
    # print(coords)
    # print()
    return coords

def remove_blank_axis(axis_bboxes, axis_text_regions, axis='row'):
    if axis == 'row':
        min_pos = 1
        max_pos = 3
    else:
        min_pos = 0
        max_pos = 2
        
    assert len(axis_bboxes) == len(axis_text_regions)
    for i in range(len(axis_bboxes)):
        if axis_text_regions[i] == []:
            axis_bboxes[i] = []
    axis_bboxes = [item for item in axis_bboxes if item != []]
    axis_bboxes = align_coords(axis_bboxes, axis=axis)
    
    for i in range(len(axis_bboxes)):
        if axis_bboxes[i][min_pos] == axis_bboxes[i][max_pos]:
            axis_bboxes[i] = []
    axis_bboxes = [item for item in axis_bboxes if item != []]
    
    return axis_bboxes

def fix_text_overlap(bboxes, word_areas, span_cells, axis='row'):
    # assert len(bboxes) == len(word_areas)
    # if len(bboxes) > 1:
    if axis == 'row':
        min_pos = 1
        max_pos = 3
    else:
        min_pos = 0
        max_pos = 2
        
    for ax in bboxes:
        for word_bbox in word_areas:
            if word_bbox[min_pos] < ax[min_pos] < word_bbox[max_pos] :
                for cell in span_cells:
                    word_in_span_iou = calculate_iou(cell, word_bbox)
                    if not word_in_span_iou > 0:
                        # if abs(ax[min_pos] - word_bbox[max_pos]) < abs(ax[min_pos] - word_bbox[min_pos]):
                        #     ax[min_pos] = word_bbox[max_pos]
                        # else:
                        ax[min_pos] = word_bbox[min_pos]
    
    return bboxes

def fix_bbox_overlaps(axis_bboxes):
    
    # for i in range(len(axis_bboxes)):
    #     if i < len(axis_bboxes)-1:
    #         iou = calculate_iou(axis_bboxes[i], axis_bboxes[i+1])
    #         if iou > 0:
    #             print(axis_bboxes[i], axis_bboxes[i+1])
        
    print()    
    for i in range(len(axis_bboxes)):
        for j in range(len(axis_bboxes)):
            if i != j:
                bbox_iou = calculate_iou(axis_bboxes[i], axis_bboxes[j])
                print(axis_bboxes[i], axis_bboxes[j], bbox_iou)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--images_path',
                        help="Path to images directory")
    parser.add_argument('--pred_path',
                        help="Path to predicted JSON files")
    parser.add_argument('--ocr_path',
                        help="Path to 'all words' OCR path")
    parser.add_argument('--out_path',
                        help="Path to store the final results")
    return parser.parse_args()

def main():
    args = get_args()
    
    images_dir = args.images_path
    pred_dir = args.pred_path
    out_dir  = args.out_path
    ocr_dir = args.ocr_path
    
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # pred_filenames = [elem for elem in os.listdir(gt_dir) if elem.endswith(".xml")]
    pred_filenames = [elem for elem in os.listdir(pred_dir) if elem.endswith('_structure.json')]
    
    
    for file_idx, filename in tqdm(enumerate(pred_filenames), 'Processing'):
        
        image_name = filename.replace('_structure.json', '.png')
        xml_name = filename.replace('_structure.json', '.xml')
        
        # ocr_filename = filename.replace('_structure.json', '.json')
        # ocr_filepath = os.path.join(ocr_dir, ocr_filename)
        
        
        # pred_filename = filename.replace('.xml', '_structure.json')
        pred_filepath = os.path.join(pred_dir, filename)
        
        pred_bboxes, pred_labels, pred_col_headers = read_json(pred_filepath)
        
        
        if not pred_bboxes == pred_labels == 0:
            
            image = cv2.imread(os.path.join(images_dir, image_name))
            
            # with open(ocr_filepath, 'r') as f:
            #     ocr_words = json.load(f)
            
            
            # masked_img = mask_all_text(image, ocr_words)
            
            
            pred_columns = [bbox for bbox, label in zip(pred_bboxes, pred_labels) if label == 'table column']
            pred_rows = [bbox for bbox, label in zip(pred_bboxes, pred_labels) if label == 'table row']
            span_cells = [bbox for bbox, label in zip(pred_bboxes, pred_labels) if label == 'table spanning cell']
            
            
            pred_rows.extend([bbox for bbox, label in zip(pred_bboxes, pred_labels) if label == 'table column header'])
            
            if not pred_columns == pred_rows == span_cells == []:
                pred_columns = sort_coord(pred_columns)
                pred_columns = align_coords(pred_columns, axis='col')
                
                pred_rows = sort_coord(pred_rows)
                pred_rows = align_coords(pred_rows)
                
                # for i in range(len(pred_rows)):
                #     get_text_area(pred_rows[i], ocr_words)
                
                # exit()
                # row_texts = [get_text_area(masked_img, pred_rows[i], ocr_words) for i in range(len(pred_rows))]
                # col_texts = [get_text_area(masked_img, pred_columns[i], ocr_words) for i in range(len(pred_columns))]
                
                # # print(row_texts)
                
                # # exit()
                # ocr_bboxes = [ocr_words[item]['bbox'] for item in ocr_words]
                
                
                # pred_rows = fix_text_overlap(pred_rows, ocr_bboxes, span_cells)
                # pred_columns = fix_text_overlap(pred_columns, ocr_bboxes, span_cells, axis='col')
                
                # pred_rows = remove_blank_axis(pred_rows, row_texts)
                # pred_columns = remove_blank_axis(pred_columns, col_texts, axis='col')
                
                # # pred_rows = fix_bbox_overlaps(pred_rows)
                # # pred_columns = fix_bbox_overlaps(pred_columns)
                
                col_header = pred_rows[:pred_col_headers]
                if col_header != []:
                    col_header = [col_header[0][:2] + col_header[-1][2:]]
                
                    all_corrected_bboxes = pred_columns + pred_rows + col_header
                    all_corrected_labels = ['table column'] * len(pred_columns) + ['table row'] * len(pred_rows) + ['table column header']
                else:
                    all_corrected_bboxes = pred_columns + pred_rows
                    all_corrected_labels = ['table column'] * len(pred_columns) + ['table row'] * len(pred_rows) 
                # print()
                # for item in row_texts:
                    
                #     # item = [item[0], item[1], item[0]+item[2], item[1]+item[3]]
                #     print(item)
                    
                #     cv2.rectangle(image, item, (244, 244, 0), 2)
                    
                # cv2.imwrite('test_pred_text.png', image)
                    
                # exit()
                
                image = Image.open(os.path.join(images_dir, image_name))
                table_w, table_h = image.size
                xml_object = generate_xml(imgname=xml_name, 
                                        table_w=table_w, 
                                        table_h=table_h, 
                                        bboxes=all_corrected_bboxes, 
                                        labels=all_corrected_labels)
                
                xml_object.write(os.path.join(out_dir, xml_name))
            # print(span_cells)
            
            
    
    
if __name__ == "__main__":
    main()