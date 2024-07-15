"""
python3 calc_stp.py --gt_path ../annotated/annotated --pred_path ../padded_pred --out_path ../results --iou_thresh 0.5 --ocr_path ../ocr

STP Calculation for TaTR model

Author: Gayathri Satheesh
Created on: Sep 19, 2023
Updated on:

Description:
Calculates Row STP, Col STP, Cell STP and Overall Table STP

Logic: 
Table STP is 1 only if Row STP and Col STP is 1

For each image, first step will be to compare the number of rows and cols predicted 
with that of actual rows and cols, only if the numbers are same, we proceed to cal-
culate the IoUs of rows and cols. If numbers are not same, the default Col STP and 
Row STP would be 0.

Row STP: If all the IoU of GT row's text region and Predicted row's text region is
         greater than specified threshold, then Row STP will be 1.
         
Col STP: If all the IoU of GT column's text region and Predicted column's text region is
         greater than specified threshold, then column STP will be 1.
"""

import os
import json

import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
from math import dist

import xml.etree.ElementTree as ET

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
    
    data = data[0]
    
    bboxes = []
    labels = []
    
    for key in data:
        for pred in data[key]:
            try: 
                # ToDo: in json, the label info for column headers is missing, update code to include that
                bboxes.append([int(item) for item in pred['bbox']])
                labels.append(pred['label'])
            except Exception as e: 
                # print(e)
                pass
        
    return bboxes, labels

def get_text_area(main_bbox, ocr_words):
    text_region = []
    
    for word in ocr_words:
        word_bbox = [int(item) for item in ocr_words[word]['bbox']]
        if calculate_iou(word_bbox, main_bbox) > 0.0:
            text_region.append(ocr_words[word]['bbox'])
    if len(text_region) > 0:
        # print(text_region)
        sorted_text_region = sort_coord(text_region)
        # print(sorted_text_region)
        x_mins = []
        y_mins = []
        x_maxs = []
        y_maxs = []
        for item in sorted_text_region:
            x_mins.append(item[0])
            y_mins.append(item[1])
            x_maxs.append(item[2])
            y_maxs.append(item[3])

        text_region_boundary = [min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)]
        return text_region_boundary
    return []

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

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gt_path',
                        help="Path to grounf truth XML files")
    parser.add_argument('--pred_path',
                        help="Path to predicted JSON files")
    parser.add_argument('--ocr_path',
                        help="Path to 'all words' OCR path")
    parser.add_argument('--iou_thresh',
                        help="Min IoU threshold to consider a predicted region as correct")
    parser.add_argument('--out_path',
                        help="Path to store the final results")
    return parser.parse_args()

def main():
    args = get_args()
    
    gt_dir = args.gt_path
    pred_dir = args.pred_path
    out_dir  = args.out_path
    ocr_dir = args.ocr_path
    iou_thresh = float(args.iou_thresh)
    
    final_data = []
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    xml_filenames = [elem for elem in os.listdir(gt_dir) if elem.endswith(".xml")]
    
    total_table_stp = 0
    total_col_stp = 0
    total_row_stp = 0
    
    
    for file_idx, filename in tqdm(enumerate(xml_filenames), 'Processing'):
        
        image_name = filename.replace('xml', 'png')
        # print(image_name)
        
        ocr_filename = filename.replace('xml', 'json')
        ocr_filepath = os.path.join(ocr_dir, ocr_filename)
        
        gt_filepath = os.path.join(gt_dir, filename)
        gt_bboxes, gt_labels = read_pascal_voc(gt_filepath)
        
        gt_columns = [bbox for bbox, label in zip(gt_bboxes, gt_labels) if label == 'table column']
        gt_rows = [bbox for bbox, label in zip(gt_bboxes, gt_labels) if label == 'table row']
        gt_rows = [bbox for bbox, label in zip(gt_bboxes, gt_labels) if label == 'table row']
        
        pred_filename = filename #filename.replace('.xml', '_structure.json')
        pred_filepath = os.path.join(pred_dir, pred_filename)
        
        pred_bboxes, pred_labels = read_pascal_voc(pred_filepath)
        
        table_stp = 0
        col_stp = 0
        row_stp = 0
        
        if not pred_bboxes == pred_labels == 0:
            
            with open(ocr_filepath, 'r') as f:
                ocr_words = json.load(f)
            
            pred_columns = [bbox for bbox, label in zip(pred_bboxes, pred_labels) if label == 'table column']
            pred_rows = [bbox for bbox, label in zip(pred_bboxes, pred_labels) if label == 'table row']
            pred_spanning_cells = [bbox for bbox, label in zip(pred_bboxes, pred_labels) if label == 'table spanning cell']
            # pred_rows.extend([bbox for bbox, label in zip(pred_bboxes, pred_labels) if label == 'table column header'])
            
            pred_columns = sort_coord(pred_columns)
            # pred_columns = align_coords(pred_columns, axis='col')
            
            pred_rows = sort_coord(pred_rows)
            # pred_rows = align_coords(pred_rows)
            
            # print(pred_rows)
            # exit()
            
            
            if len(pred_columns) == len(gt_columns):
                
                
                gt_columns = sort_coord(gt_columns)
                
                # pred_columns = rearrange_coords(gt_coords= gt_columns, pred_coords=pred_columns)
               
                # print(pred_columns)
                # exit()
                # col_ious = [round(calculate_iou(pred_columns[i], gt_columns[i]), 2) 
                #             for i in range(len(gt_columns))]
                
                is_sub = []
                col_ious = []
                for i in range(len(pred_columns)):
                    pred_col_text_region = get_text_area(pred_columns[i], ocr_words)
                    gt_col_text_region = get_text_area(gt_columns[i], ocr_words)
                    
                    if pred_col_text_region != [] and gt_col_text_region != []:
                        col_iou = calculate_iou(pred_col_text_region, gt_col_text_region)
                        col_ious.append(col_iou)
                    elif pred_col_text_region == gt_col_text_region == []:
                        col_ious.append(1)
                    else:
                        col_ious.append(0)
                    
                    # if set(gt_col_text_region).issubset(set(pred_col_text_region)):
                    #     is_sub.append(1)
                    # else:
                    #     is_sub.append(0)
                
                if min(col_ious) >= iou_thresh:
                # if 0 not in is_sub:
                    col_stp = 1
                    total_col_stp += 1
            if len(pred_rows) == len(gt_rows):
                
                
                gt_rows = sort_coord(gt_rows)
                
                # pred_rows = rearrange_coords(gt_coords= gt_rows, pred_coords=pred_rows)
                
                
                # row_ious = [round(calculate_iou(pred_rows[i], gt_rows[i]), 2) 
                #             for i in range(len(gt_rows))]
                # if min(row_ious) >= iou_thresh:
                #     row_stp = 1
                #     total_row_stp += 1
                
                row_ious = []
                for i in range(len(pred_rows)):
                    pred_row_text_region = get_text_area(pred_rows[i], ocr_words)
                    gt_row_text_region = get_text_area(gt_rows[i], ocr_words)
                    if pred_row_text_region != [] and gt_row_text_region != []:
                        row_iou = calculate_iou(pred_row_text_region, gt_row_text_region)
                        row_ious.append(row_iou)
                    elif pred_row_text_region == gt_row_text_region == []:
                        row_ious.append(1)
                    else:
                        row_ious.append(0)
                    
                    # if set(pred_col_text_region).issubset(set(pred_row_text_region)):
                    #     is_sub.append(1)
                    # else:
                    #     is_sub.append(0)
                
                if min(row_ious) >= iou_thresh:
                # if 0 not in is_sub:
                    row_stp = 1
                    total_row_stp += 1
                    
            if col_stp == row_stp == 1:
                table_stp = 1
                total_table_stp += 1
            
        final_data.append({'n_item': file_idx+1,
                           'image_name': image_name,
                           'Row STP': row_stp,
                           'Col STP': col_stp,
                           'Table STP': table_stp})
    # exit()
    final_data.append({'image_name': 'OVERALL',
                        'Row STP': total_row_stp,
                        'Col STP': total_col_stp,
                        'Table STP': total_table_stp})
        
    pd.DataFrame(final_data).to_csv(os.path.join(out_dir, 'final_report.csv'), index=False)    
    
    
if __name__ == "__main__":
    main()