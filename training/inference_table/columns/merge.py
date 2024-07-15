import os 
import json

import itertools

import sys
sys.path.append('../')
import utils
import copy

import numpy as np

import cv2

from fitz import Rect

from tqdm import tqdm

import configparser
# config=configparser.ConfigParser()
# config.read('config.ini')

# col_cords_path = config['PATH']['SAVE_COORDS_DETECT_COORDS']
# header_coords_path = config['PATH']['HEADER_COORDS']
# img_path = config['PATH']['IMAGE_FOLDER']



# coords_list = os.listdir(col_cords_path)

# save_path = config['PATH']['FINAL_COORDS_SAVE_PATH']

# os.makedirs(save_path, exist_ok=True)

# col_viz_path = config['PATH']['FINAL_IMAGE_SAVE_PATH']
# os.makedirs(col_viz_path, exist_ok=True)

# with open('single_line_elements.json', 'r') as f:
#     single_line_elements = json.load(f)

# single_line_elements= single_line_elements['headers']

# single_line_elements = [item.lower() for item in single_line_elements]

# class ImageProps:
#     def __init__(self,json_path, img_path) -> None:
#         json_name = json_path.split('/')[-1]
#         self.image_name = json_name.replace('json','png')
#         ocr_path = config['PATH']['OCR_PATH']
#         self.imgfile = os.path.join(img_path, self.image_name)
#         header_strip_path = config['PATH']['HEADER_STRIP']
#         with open(os.path.join(ocr_path, json_name), 'r') as f:
#             self.ocr_data = json.load(f)
#         with open(header_strip_path, 'r') as f:
#             self.header_strip_data = json.load(f)
#         self.header_present = False
#         if self.image_name in self.header_strip_data.keys():
#             self.header_data = self.header_strip_data[self.image_name]['final_header']
#             self.header_present = True
        
#     def is_empty_gap(self, gap):
       
#         return utils.get_text(gap, self.ocr_data) == []


#     def header_texts(self):
#         words = utils.get_text(self.header_data, self.ocr_data)
#         return words#[word['word'] for word in words]
    
#     # def gap_threshold(self):
#     #     header_width_in_px = self.header_data[2] - self.header_data[0]
        
#     #     header_string = ''.join(self.header_texts())
#     #     n_chars_in_header = len(header_string)
#     #     px_threshold = 3
        
#         return ((header_width_in_px/n_chars_in_header)*px_threshold) * 1.5#-3
        
#     def get_text(self, region):
#         return utils.get_text(region, self.ocr_data)
#         # bboxes = [word['bbox'] for wo]
#         # return text
    
#     def get_bbox_in_region(self, region):
#         texts = self.get_text(region)
#         return utils.sort_coord([item['bbox'] for item in texts])
        
#     def get_content(self, region):
#         contents = self.get_text(region)
#         return ' '.join([item['word'] for item in contents])
    
#     def line_overlapping_header_cell(self, col_x, header_cells):
#         return any(col_x >= cell[0]  and cell[2] >= col_x for cell in header_cells)
    
#     # def gap_exceeds_threshold(self, gap_coords):
#     #     return gap_coords[0]-gap_coords[3] > self.gap_threshold()
    
#     def is_line_overlapping_text(self, pt, region):
#         region_text =  self.get_text(region)
#         text_bboxes = [item['bbox'] for item in region_text]
#         if text_bboxes != []:
#             overlaps = [bbox[0] <= pt and bbox[2] >= pt for bbox in text_bboxes]   
#             # n_overlaps = [1 for i in overlaps]
#             overlaps_perc = sum(1 for i in overlaps if i == True)/len(overlaps)   
                    
#             return overlaps_perc, any(overlaps)
#         else:
#             return 0, False
    
#     def get_header_line_intersection(self, line_coords, cells):
#         pt = line_coords[0]
#         for cell in cells:
#             if self.line_overlapping_header_cell(pt, [cell]):
#                 return cell

def get_content(region, ocr_data):
    contents = utils.get_text(region, ocr_data)
    return ' '.join([item['text'] for item in contents])

def get_bbox_in_region(region, ocr_data):
    texts = utils.get_text(region, ocr_data)
    return utils.sort_coord([item['bbox'] for item in texts])

def line_overlapping_header_cell(col_x, header_cells):
        return any(col_x >= cell[0]  and cell[2] >= col_x for cell in header_cells)
    
def get_header_line_intersection(line_coords, cells):
    pt = line_coords[0]
    for cell in cells:
        if line_overlapping_header_cell(pt, [cell]):
            return cell
def get_lines_within_gap(col, cell_gap):
    lines = []
    idx = []
    for i in (range(len(col))): 
        # if not col[i][1] == 'solid_lines':
        # #print(col, cell_gap)
        if col[i][0][0] >= cell_gap[0] and col[i][0][0] <= cell_gap[2]:
            lines.append(col[i])
            idx.append(i)
    
    return idx, lines

def is_line_overlapping_text(pt, region, ocr_data):
    region_text =  utils.get_text(region, ocr_data)
    text_bboxes = [item['bbox'] for item in region_text]
    if text_bboxes != []:
        overlaps = [bbox[0] <= pt and bbox[2] >= pt for bbox in text_bboxes]   
        # n_overlaps = [1 for i in overlaps]
        overlaps_perc = sum(1 for i in overlaps if i == True)/len(overlaps)   
                
        return overlaps_perc, any(overlaps)
    else:
        return 0, False
    
def get_column_region(lines_idxs, cols):
   lines_idxs.insert(0,lines_idxs[0]-1) 
   lines_idxs = lines_idxs[:-1]
   return [[cols[i][0][0], cols[i][0][1], cols[i+1][0][2], cols[i+1][0][3]]
                    for i in lines_idxs]

def get_line(cell_gap, col_y_min, col_y_max, ocr_data):
    col_region = [cell_gap[0], col_y_min, cell_gap[2], col_y_max]
    cell_width = cell_gap[2] - cell_gap[0]
    # if cell_width < int(imgprops.gap_threshold()):
    #     return None
    # gap_width = (cell_gap[2]-cell_gap[0])
    overlap_percentanges = []
    for pt in range(cell_gap[2], cell_gap[0], -1):
        n_overlap, line_overlaps =  is_line_overlapping_text(pt, col_region, ocr_data)
        
        overlap_percentanges.append( n_overlap ) #[n_overlap, len(n_overlap )])
        if not line_overlaps:  
            return [pt, col_y_min ,pt, col_y_max]
    
    
    # if len(count_overlaps) == 1:
    #     return [pt, col_y_min ,pt, col_y_max]
    
    if overlap_percentanges != []:
        overlap_percs = overlap_percentanges
        # overlap_percs = [item[0].count(True)/item[1] for item in overlap_percentanges]
        min_overlap_perc = min(overlap_percs) 
        cell_gaps = [i for i in range(cell_gap[2], cell_gap[0], -1)]
        
        if min_overlap_perc <= 0.5:
            min_idx = overlap_percs.index(min_overlap_perc)
            return [cell_gaps[min_idx], col_y_min ,cell_gaps[min_idx], col_y_max]
        
        # try:
        return [cell_gaps[len(cell_gaps)-1], col_y_min ,cell_gaps[len(cell_gaps)-1], col_y_max]
        # except Exception:
        #     if len(cell_gaps) > 5:
        #         return [cell_gaps[0], col_y_min ,cell_gaps[0], col_y_max]
        # least_overlaps = max(overlap_percentanges)
        # max_idx = overlap_percentanges.index(least_overlaps)
        # return [cell_gaps[max_idx], col_y_max ,cell_gaps[max_idx], col_y_max]
        

def get_new_line(cell_gap, col, image_h, ocr_data, start_from = None):
    if col == []:
        col_y_min = 0
        col_y_max = image_h
    else:    
        col_y_min = col[0][0][1]
        col_y_max = col[0][0][3]
    if start_from == 'mid':
        cell_gap = [cell_gap[0], cell_gap[1], int(cell_gap[2]-cell_gap[0]), cell_gap[3]]
    line = get_line(cell_gap, col_y_min, col_y_max, ocr_data)
    # #print(line)
    return None if line is None else [line, '']

def get_header_and_col_text(col_region, header_data, ocr_data):
    header_text = utils.get_text([col_region[0], header_data[1] ,col_region[2], header_data[3]], ocr_data)
    column_text = utils.get_text([col_region[0], header_data[3]+1 ,col_region[2], col_region[3]], ocr_data)
    header_text = [item['text'] for item in header_text]
    column_text = [item['text'] for item in column_text]
    return header_text, column_text

def keep_col(cell_gap, line_idxs, lines_in_gap, col, header_strip, image_h, ocr_data):
    # #print(f"Cell gaps, lines in gap: {cell_gap, lines_in_gap}")
    if len(lines_in_gap) == 1:
        return lines_in_gap[0]
    elif len(lines_in_gap) > 1:
        col_regions = get_column_region(line_idxs,  col)
        for i in range(len(col_regions)-1):
            first_header_text, first_column_text = get_header_and_col_text(col_regions[i], header_strip, ocr_data)
            next_header_text, next_column_text = get_header_and_col_text(col_regions[i+1], header_strip, ocr_data)
            # #print(first_header_text, first_column_text)
            # #print(next_header_text, next_column_text)
            # #print()
            
            if len(first_header_text) >= 1 and len(first_column_text) == 0 and \
                len(next_header_text) == 0 and len(next_column_text) >= 1 and \
                    lines_in_gap[i][1] == 'padded_lines':
                    
                    return lines_in_gap[i+1]
            # elif len(first_header_text) >= 1 and len(first_column_text) >= 1 and \
            #     len(next_header_text) == 0 and len(next_column_text) >= 1 and \
            #         lines_in_gap[i][1] == 'padded_lines':
                        
            #             return lines_in_gap[i]
            # elif len(first_header_text) >= 1 and len(first_column_text) >= 1 and \
            #     len(next_header_text) == 0 and len(next_column_text) >= 1 and \
            #         lines_in_gap[i][1] == 'padded_lines':
            #     # #print( first_header_text, first_column_text )
            #     # #print()
            #     # #print(next_header_text, next_column_text)
            #     # exit()
            #     return lines_in_gap[i]
        # if (cell_gap[2] - cell_gap[0])+10 >= imgprops.gap_threshold():
        #     col_y_min = col[0][0][1]
        #     col_y_max = col[0][0][3]
        #     line = get_line(cell_gap, imgprops, col_y_min, col_y_max)
        #     return None if line is None else [line, '']
        # return None
    elif len(lines_in_gap) == 0:
        return get_new_line(cell_gap, col, image_h, ocr_data)
        

def line_already_present(line, cols, buffer=9):
    line_start_buffer = line[0]-buffer
    line_end_buffer = line[0]+buffer
    
    for col in cols:
        if (line_start_buffer <= col[0][0]) \
        or (line_start_buffer >= col[0][0] and col[0][0] <= line_end_buffer):
            return True
    return False


def find_line_space(cell_gap, cols, image_h, ocr_data):


    # blank_image = np.full((cell_gap[2]-cell_gap[0], image_h), 255, dtype=int)
    gap_region = [cell_gap[0], cell_gap[1], cell_gap[2], image_h]

    area_ocr = get_bbox_in_region(gap_region, ocr_data)

    word_x = []
    for word in area_ocr:
        word_x.append(word[0])
        word_x.append(word[2])
    if not word_x == []:
        min_text_point = min(word_x)# - cell_gap[0]
        max_text_point = max(word_x)# - cell_gap[0]

        cols = []
        if min_text_point > cell_gap[0]:
            cell_gap = [cell_gap[0], cell_gap[1], min_text_point, cell_gap[3]]
            
        elif max_text_point > cell_gap[2]:
            cell_gap = [cell_gap[2], cell_gap[1], max_text_point, cell_gap[3]]

        line = get_new_line(cell_gap, cols, image_h, ocr_data)
    else:
        line = None
    
    return line



def header_col_compare(cols, cells, header_strip, image_h, ocr_data, header_dict):
    """
    """
    
    # cells = [header_cells[cell] for cell in header_cells]
    # cells = header_cells
    
    cells = utils.sort_coord(cells)
    cell_gaps = [[cells[i][2], min(cells[i][1],cells[i+1][1]), cells[i+1][0], max(cells[i][3],cells[i+1][3])] 
                 for i in range(len(cells)-1)]
    
    
    cols = [col for col in cols 
                if not col[0][0] <= header_strip[0]+10 and not col[0][0] >= header_strip[2]-10]
    
    # if cols == []: return None
    
    final_lines = []
    
    for col in cols:
        if col[1] == 'solid_lines': 
            final_lines.append(col)
    
    
    for cell_gap in cell_gaps:
        line_idxs, lines_in_gap = get_lines_within_gap(cols, cell_gap)

        # if lines_in_gap == []:
            
        #     corrected_col = keep_col(cell_gap, line_idxs, lines_in_gap, cols, imgprops)
        #     if not corrected_col == None or corrected_col == []:
        #         final_lines.append(corrected_col)
        
        if len(lines_in_gap) > 1:
            line_types = [line[1] for line in lines_in_gap]
            if not 'solid_lines' in line_types:
                line = get_new_line(cell_gap, cols, image_h, ocr_data) #, start_from='mid')
                        # exit("EXITING")
                        # #print(f"Line: {line}")
                if line is not None:
                    line = line[0]
                    final_lines.append([line,'padded_line'])
        elif len(lines_in_gap) == 1:
            final_lines.extend(lines_in_gap)
        else:
            line = find_line_space(cell_gap, cols, image_h, ocr_data)
            if line is not None:
                line = line[0]
                final_lines.append([line,''])
            
        # else:
        #     final_lines.extend(lines_in_gap)
            
        # if len(lines_in_gap) > 1:
        #     corrected_col = keep_col(cell_gap, line_idxs, lines_in_gap, cols, header_strip, image_h, ocr_data)
        #     if corrected_col != None:
        #         final_lines.append(corrected_col)
        # elif len(lines_in_gap) == 0:
        #     line = get_new_line(cell_gap, cols, image_h, ocr_data) #, start_from='mid')
        #         # exit("EXITING")
        #         # #print(f"Line: {line}")
        #     if line is not None:
        #         line = line[0]
        #         final_lines.append([line,''])
        # else:
        #     final_lines.extend(lines_in_gap)
                
    # col_y_min = cols[0][0][1]
    # col_y_max = cols[0][0][3]
    # gap_threshold = imgprops.gap_threshold()
    
    # for cell_gap in cell_gaps:
    #     line_idxs, lines_in_gap = get_lines_within_gap(final_lines, cell_gap)
        
    #     if lines_in_gap == [] or lines_in_gap is None:
    #         # #print("hhhhhhhh")
    #         # #print(cell_gap[2]-cell_gap[0] , int(gap_threshold))
    #         # if (cell_gap[2]-cell_gap[0]) >= int(gap_threshold):
    #             # #print(cell_gap, gap_threshold)
    #             # exit()
    #             line = get_new_line(cell_gap, cols, image_h, ocr_data) #, start_from='mid')
    #             # exit("EXITING")
    #             # #print(f"Line: {line}")
    #             if line is not None:
    #                 line = line[0]
    #                 final_lines.append([line,''])
        
                
    # #print(final_lines)
    
    
    
    if final_lines != []:
        header_xmin = header_strip[0]
        header_xmax = header_strip[2]
        if final_lines[0][0][0] > header_xmin:
            start_line = [0, 0, 0, final_lines[0][0][3]]
            # if not line_already_present(start_line, final_lines):
            final_lines.insert(0, [start_line, ''])
        if final_lines[-1][0][0] < header_xmax:
            end_line = [header_xmax, 0, header_xmax, final_lines[0][0][3]]
            # if not line_already_present(end_line, final_lines):
            final_lines.append([end_line, ''])
            

        """
        ToDo: 
        bottom section
        draw a line only if there is some conntent in the top 50% of the column except header
        """
            
        all_cols = [item[0] for item in cols]
        all_final_lines = [item[0] for item in final_lines]            
        for line in all_cols:
            intersecting_cell = get_header_line_intersection(line, cells)

            if line not in all_final_lines and  intersecting_cell != None:

                intersecting_cell_content = get_content(intersecting_cell, ocr_data).strip().lower()

                if intersecting_cell_content not in header_dict:
                
                    slice_1_width = line[2] - intersecting_cell[0]
                    slice_2_width = intersecting_cell[2] - line[2]
                    
                    if not slice_1_width <= 10 and not slice_2_width <= 10:
                        slice_1_header = [intersecting_cell[0], intersecting_cell[1], line[2], intersecting_cell[3]]
                        slice_2_header = [line[2], intersecting_cell[1], intersecting_cell[2], intersecting_cell[3]]
                        slice_1 = [intersecting_cell[0], header_strip[3]+2, line[2], line[3]]
                        slice_2 = [line[2], header_strip[3]+2, intersecting_cell[2], line[3]]
                        
                        text_bbox_in_slice_1_header = utils.get_merged((get_bbox_in_region(slice_1_header, ocr_data)))
                        text_bbox_in_slice_2_header = utils.get_merged((get_bbox_in_region(slice_2_header, ocr_data)))
                        
                        text_bbox_in_slice_1 = utils.get_merged((get_bbox_in_region(slice_1, ocr_data)))
                        text_bbox_in_slice_2 = utils.get_merged((get_bbox_in_region(slice_2, ocr_data)))
                    
                        if text_bbox_in_slice_1_header != [] \
                            and text_bbox_in_slice_2_header != [] \
                            and text_bbox_in_slice_1 != [] \
                            and text_bbox_in_slice_2 != []:
                            final_lines.append([line, '']) 

        return final_lines
    
    # #print()
    
            
    return cols
    

def get_sorted_coords(col_dict):
    if col_dict['padded_lines'] == [] and col_dict['solid_lines'] == []:
        return col_dict
    for key in col_dict:
        if key=='padded_lines':
            col_dict[key] = col_dict[key][0]
        for i in range(len(col_dict[key])):
            col_dict[key][i][0].extend(col_dict[key][i][1])
            col_dict[key][i].pop(1)
            col_dict[key][i] = col_dict[key][i][0]
            
    cols = copy.deepcopy(col_dict)
    cols['padded_lines'].extend(cols['solid_lines'])
    
    cols = cols['padded_lines']
    # cols = [item[0]+item[1] for item in cols]
    cols = utils.sort_coord(cols)
    
    for i in range(len(cols)):
        if cols[i] in col_dict['padded_lines']:
            cols[i] = (cols[i], 'padded_lines')
        elif cols[i] in col_dict['solid_lines']:
            cols[i] = (cols[i], 'solid_lines')
    
    return cols



def split_columns(cols, cells, imgprops):
    gap_region = [[cells[i][2], cells[i][3], cells[i+1][0], cells[i+1][1]] 
                 for i in range(len(cells)-1)]
    columns = []
    for gap in gap_region:
        if imgprops.is_empty_gap(gap):
            cols = [[cols[0], cols[1], gap[2] ,cols[3]],
                    [gap[2], cols[1], cols[2], cols[3]]]
            columns.extend(cols)
    return columns

def remove_overlaps(col_coords, header_data, ocr_data):
    
    ocr_coords = utils.get_text(header_data, ocr_data)
    ocr_coords = [item['bbox'] for item in ocr_coords]
    # #print(ocr_coords)
    
    for i, col in enumerate(col_coords):
        if col[1] == 'padded_lines':
            for coord in ocr_coords:
                if (coord[0] <= col[0][1] <= coord[2]) and (col[0][1] != header_data[0]):
                    
                    col_coords[i] = None
        
    return [item for item in col_coords if item is not None]

def modify_columns(image, column_coordinate, header_coordinate): 
    pass

def get_corrected_columns(col_data, header_cells, image_h, header_strip, ocr_data, single_line_elements, header_dict):   
    
    if len(col_data['solid_lines']) == len(header_cells) == len(col_data['padded_lines'][0]) == 0 :
        return {'single_line_elements': [],
                    'column_seperators': [],
                    'meta': []}
    col_coords = get_sorted_coords(col_data)


    # print(f"Header cells: {header_cells}" )
    col_updated_coords = col_coords
    if header_cells != [] and len(col_updated_coords) > 1:
        col_updated_coords = header_col_compare(col_coords, header_cells, header_strip, image_h, ocr_data, header_dict)
        col_updated_coords = remove_overlaps(col_updated_coords, header_strip, ocr_data)
    
    cells_with_single_line = []
    
    if col_updated_coords != []:
        line_coords = []
        for item in col_updated_coords:
            if item[0] not in line_coords:
                line_coords.append(item[0])
        
        
        col_y_min = line_coords[0][1]
        col_y_max = line_coords[0][3]
        line_coords = [item[:2] for item in line_coords]
        line_coords = utils.sort_coord(line_coords)
        
        column_coordinates = [
            [line_coords[i][0], col_y_min, line_coords[i+1][0], col_y_max]
            for i in range(len(line_coords)-1)
        ]
        
    
        anchor_words = ['unit', 'rate', 'amt', 'amount', 'quantity', 'qty']
        if header_strip != None:
            header_cells = [Rect.intersect(Rect(header_strip), Rect(col))
                        for col in column_coordinates]
            
            for i in range(len(header_cells)):
                cell_contents = get_content(header_cells[i], ocr_data) #.split()
                
                # cell_contents = [item.lower() for item in cell_contents]
                # for item in cell_contents:
                #     #print(item)
                if cell_contents in single_line_elements and i not in cells_with_single_line:
                        cells_with_single_line.append(i)
                elif i not in cells_with_single_line:
                    for wrd in anchor_words:
                        if cell_contents.startswith(wrd):
                            cells_with_single_line.append(i)
                            break
            # #print(cells_with_single_line)
        else:
            header_cells = []

        final_result = {'single_line_elements': cells_with_single_line,
                        'column_seperators': col_updated_coords,
                        'meta': []}
        for i in range(len(header_cells)):
            header_cell_coordinate = []
            if not header_cells == []:
                header_cell_coordinate = [int(pt) for pt in header_cells[i]]
            cell_content = get_content(header_cell_coordinate, ocr_data)
            # print(f"Cell content: {cell_content}")
            column_without_header = [column_coordinates[i][0], header_cell_coordinate[3]+1,
                                    column_coordinates[i][2], column_coordinates[i][3]]
            final_result['meta'].append({
                'header_cell_content': cell_content,
                'header_cell_coordinate': header_cell_coordinate,
                'full_column_coordinate': column_coordinates[i],
                'column_without_header': column_without_header
                })
        return final_result
            
    else:
        return {'single_line_elements': cells_with_single_line,
                'column_seperators': col_updated_coords,
                'meta': []}