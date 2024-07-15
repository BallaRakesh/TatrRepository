import os 
import cv2
import json
import utils
import numpy as np
from math import dist as px_dist
from itertools import groupby
from operator import itemgetter
import tqdm

def binarize(img):
    otsu_threshold, image_res = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY +  cv2.THRESH_OTSU)
    return otsu_threshold, image_res

def invert_image(img):
    pass

def is_white_bg(img, threshold):
    
    h, w = img.shape
    edge_colors = [img[0][0], img[0][w-1], img[h-1][w-1], img[h-1][0], 255]
    if len(set(edge_colors)) <= 2:
        return True
    return False

def get_min_max(data, drop_k=2):
    
    pixel_set =  [data[item] for item in data if len(data[item]) > drop_k]
    
    pixel_set_list = [item for sublist in pixel_set for item in sublist]
    # max_list = max(data.values(), key=len)
    if pixel_set_list != []:
        return min(pixel_set_list), max(pixel_set_list)
    return 0, 0
    
def continous_pixels(image_array, axis=0):
    pixels = sorted(list(set(image_array[:,axis])))
    
    grouped_pixels = {}    
    i = 0
    for k, g in groupby(enumerate(pixels), lambda x: x[1] - x[0]):
        grouped_pixels[i] = list(map(itemgetter(1), g))
        i += 1
    
    # min_coord, max_coord = get_min_max(grouped_pixels)
    return grouped_pixels

def eliminate_pixels(image_array, threshold, axis=0):
    h, w = image_array.shape
    image_aspect = h
    if axis == 0:
        image_aspect = w
    coords = np.column_stack(np.where(image_array < threshold))
    pixel_stack = {px[axis]:0 for px in coords}
    for px in coords:
        pixel_stack[px[axis]] += 1
    
    black_pxs_stack = {blacks : round((pixel_stack[blacks]/image_aspect)*100, 2) for blacks in pixel_stack}                    
    
    remove_index = [px for px in black_pxs_stack if black_pxs_stack[px] >= 90 or black_pxs_stack[px] <= 2 ] 
     
    updated_coords = np.asarray([px for px in coords if px[axis] not in remove_index])
    
    return updated_coords

def correct_bbox(ocr_data, image):
    threshold, binarized_image = binarize(image)
    for word in ocr_data:
        
            min_coords, max_coords = ocr_data[word]['vertices'][0], ocr_data[word]['vertices'][2]
            word_image = binarized_image[min_coords[1]:max_coords[1], min_coords[0]:max_coords[0]]
            
            xmin, ymin = min_coords
            xmax, ymax = max_coords
            
            vertical_lines_updated_coords = eliminate_pixels(word_image, threshold, axis=1)
            
            if vertical_lines_updated_coords.size != 0:
                
                word_xmin = min(vertical_lines_updated_coords[:,1])
                word_xmax = max(vertical_lines_updated_coords[:,1])
                grouped_pixels_x = continous_pixels(vertical_lines_updated_coords, axis=1)
                
                word_xmin, word_xmax = get_min_max(grouped_pixels_x)
                xmin = min_coords[0] + word_xmin - 1
                xmax = min_coords[0] + word_xmax + 1
                
                word_image= word_image[0:word_image.shape[0], word_xmin:word_xmax]
            
            horizontal_lines_updated_coords = eliminate_pixels(word_image, threshold, axis=0)
            if horizontal_lines_updated_coords.size != 0:
            
                word_ymin = min(horizontal_lines_updated_coords[:,0])
                word_ymax = max(horizontal_lines_updated_coords[:,0])
                
                grouped_pixels_y = continous_pixels(horizontal_lines_updated_coords, axis=0)
            
            
                word_ymin, word_ymax = get_min_max(grouped_pixels_y)
                
                
                ymin = min_coords[1] + word_ymin - 1
                ymax = min_coords[1] + word_ymax + 1
                word_image= word_image[word_ymin:word_ymax,0:word_image.shape[1]]
        
            
            
            ocr_data[word]['vertices'] = [[int(xmin), int(ymin)], 
                                        [int(xmax), int(ymin)], 
                                        [int(xmax), int(ymax)], 
                                        [int(xmin), int(ymax)]]
    return ocr_data


if __name__ == '__main__':
    image_source = '/home/gayathri/table_processing/data/input/table_images/images'
    image_list = os.listdir(image_source)

    ocr_source = '/home/gayathri/table_processing/data/input/table_images/ocr'
    ocr_list = os.listdir(ocr_source)

    save_to = '/home/gayathri/table_processing/data/input/table_images/corrected_ocr'
    os.makedirs(save_to, exist_ok=True)

    for item in tqdm.tqdm(image_list):
        image = cv2.imread(os.path.join(image_source, item), 0)

        jsonname = item.replace('png','json')
        with open(os.path.join(ocr_source, jsonname), 'r') as f:
            ocr_data = json.load(f)

        updated_ocr = correct_bbox(ocr_data=ocr_data, image=image)

        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR )
        for word in updated_ocr:
            cv2.rectangle(color_image, updated_ocr[word]['vertices'][0], updated_ocr[word]['vertices'][2], (122, 122, 0), 2)

        cv2.imwrite(f'ocr_abbyy_grasim/{item}', color_image)

        with open(os.path.join(save_to, jsonname), 'w') as f:
            json.dump(updated_ocr, f, indent=4)