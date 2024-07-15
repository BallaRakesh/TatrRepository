import configparser
from google.cloud import vision_v1p4beta1 as vision
from google.protobuf.json_format import MessageToDict
import cv2
import numpy as np
import os
import shutil
import copy
import json
from tqdm import tqdm
import time
from itertools import groupby
from operator import itemgetter

import sys
sys.path.append('../')
import utils

import logging

def get_line_color(pixel_strip, pixel_count, threshold=90, return_thresold=False):
    white_pxs = np.count_nonzero(pixel_strip)
    white_px_qty = (white_pxs / pixel_count)*100
    if white_px_qty >= threshold:
        if return_thresold:
            return 1, white_px_qty
        return 1
    if return_thresold:
        return 0, white_px_qty
    return 0


def continous_pixels(image_array=None, axis=0, pixels=None):
    
    if pixels==None:
        pixels = sorted(list(set(image_array[:,axis])))
    
    grouped_pixels = {}    
    i = 0
    for k, g in groupby(enumerate(pixels), lambda x: x[1] - x[0]):
        grouped_pixels[i] = list(map(itemgetter(1), g))
        i += 1
    
    # min_coord, max_coord = get_min_max(grouped_pixels)
    return grouped_pixels


def process_image(annotation, image):
    plain_image = np.full(image.shape[:2], 255, dtype=np.uint8)
    

    for word in annotation:
        xmin = annotation[word]['vertices'][0][0]
        xmax = annotation[word]['vertices'][2][0]

        ymin = 0
        ymax = image.shape[0]
        cv2.rectangle(plain_image, [xmin, ymin] , [xmax, ymax] , (0, 0, 0), -1)

    lines = []
    pixels = []
    plain_image_copy = copy.deepcopy(plain_image)
    plain_image_transpose = cv2.rotate(plain_image_copy, cv2.ROTATE_90_CLOCKWISE)
    for loc, hr_px in enumerate(plain_image_transpose):
        # line_color = get_line_color(hr_px, pixel_count = image.shape[1], threshold=40)
        # if line_color == 0:
        #     if (loc/image.shape[0])*100 > 0.1 and seperators['header_seperator'] == None:
        #         seperators['header_seperator'] = (0, loc, image.shape[1], loc)
        #     seperators['horizontal_lines'].append((0, loc, image.shape[1], loc))
        
        # print(hr_px[int(hr_px.size*0.8):])
        # cv2.imwrite('testt.png', hr_left_right_cropped)
        white_line_color = get_line_color(hr_px, pixel_count = plain_image_transpose.shape[1], threshold=99.8)
        
        if white_line_color == 1:
            # if seperators['horizontal_lines'] == []:
                plain_image_transpose[loc] = 0
                lines.append([loc, 0, loc, plain_image_transpose.shape[1]])
                pixels.append([loc, 0])
    cv2.imwrite('tttttest.png', plain_image_transpose)
    cv2.imwrite('TEST.png', plain_image)
    # grouped_pixels = continous_pixels(pixels, axis=0)
    # print(grouped_pixels)

    exit()

    return {'solid_lines': [], 'padded_lines':[lines]}

def detect_column(annotations, image):
    column_line_coordinates = None
    final_column_coordinate = None
    
    if not annotations == None:
        original_image = copy.deepcopy(image)

        lines_data = process_image(annotations, original_image)
    

        print(lines_data)
        return lines_data


