import os 
import json
import utils

def is_rotated(ocr_data):
    """
    Checks if the image is rotated by 180 degrees
    """
    if type(ocr_data) == dict:
        ocr_data = list(ocr_data.values())
    
    sorted_coords = utils.sort_by_y(ocr_data)[:4]
    orientation_shift = []
    
    for item in sorted_coords:
        if item['vertices'][0][0] >  item['vertices'][2][0]:
            orientation_shift.append(1)
    
    if len(orientation_shift) == len(sorted_coords) and set(orientation_shift) == {1}:
        return True
    
    return False
    

    
    