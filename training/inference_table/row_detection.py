import os 
import cv2
import numpy as np
from itertools import groupby
from operator import itemgetter


def get_areas(image, white_pixels=None, black_pixels=None, main_bg=1, header_bg=1):
    
    # print(white_pixels)
    seperators = []
    
    if main_bg == header_bg == 1:
        # inverted_image = verify_image_top(image, header_bg, main_bg)
        seperators, new_image = get_all_seperator(image)
    else:
        # inverted_image = invert_image(image, header_bg)
        inverted_image = invert_image(image, header_bg)
        seperators, new_image = get_all_seperator(inverted_image)
    
    return seperators, new_image


def verify_image_top(image, header_bg, main_bg):
    image_top = image[0:int(image.shape[0]*0.3), 0:image.shape[1]]

    for hr_px in range(image_top.shape[0]):
        n_black = np.count_nonzero(image_top[hr_px])/image_top.shape[0]
        if n_black > 0.9:
            image[hr_px,:] = 255
    return image
            

def invert_image(image, header_bg):
    locs = []
    for loc, hr_px in enumerate(image):
        line_color = get_line_color(hr_px, pixel_count = image.shape[0])
        if line_color == header_bg:
            locs.append(loc)
    if not locs == []:
        locs = sorted(locs)
        continous = []
        for i in locs:
            if len(continous) == 0:
                continous.append(i)
            elif i - continous[len(continous)-1] <= 10:
                continous.append(i)
        header_region = [0, min(continous), image.shape[1], max(continous)]
        if not header_region[0] == header_region[2] and not header_region[1] == header_region[3]:
            header_inverted = cv2.bitwise_not(image[header_region[1]:header_region[3], header_region[0]:header_region[2]])
        # try:
        
            image[header_region[1]:header_region[3], header_region[0]:header_region[2]] = header_inverted
        # except:
        #     return image
    return image

def get_all_seperator(image, line_color = 1):
    
    seperators = {
                    'horizontal_lines' : [],
                    'vertical_lines' : [],
                    'header_seperator' : None
                }
    lines_info = []
    image_left = image[0:image.shape[0], 0:int(image.shape[1]*0.001)]
    image_right = image[0:image.shape[0], int(image.shape[1]*0.85):image.shape[1]]
    image_edges = np.concatenate([image_left, image_right], axis=1)
    
    hr_locs = []
    for loc, hr_px in enumerate(image_edges):
        # line_color = get_line_color(hr_px, pixel_count = image.shape[1], threshold=40)
        # if line_color == 0:
        #     if (loc/image.shape[0])*100 > 0.1 and seperators['header_seperator'] == None:
        #         seperators['header_seperator'] = (0, loc, image.shape[1], loc)
        #     seperators['horizontal_lines'].append((0, loc, image.shape[1], loc))
        
        # print(hr_px[int(hr_px.size*0.8):])
        # cv2.imwrite('testt.png', hr_left_right_cropped)
        white_line_color = get_line_color(hr_px, pixel_count = image_edges.shape[1], threshold=95)
        
        if white_line_color == 1:
            # if seperators['horizontal_lines'] == []:
                hr_locs.append(loc)
                
            # elif loc -  seperators['horizontal_lines'][len(seperators['horizontal_lines'])-1][1] >= 10:
            #     seperators['horizontal_lines'].append((0, loc, image.shape[1], loc))
    # print(lines_info)
    # exit()
                
    # image = remove_all_black(image)

    # cv2.imwrite('img.png', image)
    vr_locs = []                
    image_transpose = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    for loc, vr_px in enumerate(image_transpose):
        line_color = get_line_color(vr_px, pixel_count = image.shape[0], threshold=65)
        if line_color == 0:
            image[:,loc] = 255 #hide the black line 
            image_transpose[loc,:] = 255
            # seperators['vertical_lines'].append((loc, 0, loc, image.shape[0]))

    # cv2.imwrite('image_transpose.png', image_transpose)
    seperators['row_bboxes'] = []
    for i in range(len(hr_locs)-2):
        if hr_locs[i] + 4 < hr_locs[i+1]:
            if len(seperators['horizontal_lines']) != 0:
                # print(len(seperators['horizontal_lines'])-1, )
                if hr_locs[i] - seperators['horizontal_lines'][len(seperators['horizontal_lines'])-1][1] > 5 :
                    prev_line = seperators['horizontal_lines'][len(seperators['horizontal_lines'])-1]
                    if np.count_nonzero(image[hr_locs[i]]) > 0:
                        
                        new_loc = get_non_zero_loc(image, hr_locs[i], threshold=0.99)
                        seperators['horizontal_lines'].append((0, new_loc, image.shape[1], new_loc))
                        seperators['row_bboxes'].append([0, prev_line[3], image.shape[1], new_loc])
                    else:
                        seperators['horizontal_lines'].append((0, hr_locs[i]-1, image.shape[1], hr_locs[i]-1))
                        seperators['row_bboxes'].append([0, prev_line[3], image.shape[1], hr_locs[i]-1])
            else:
                seperators['row_bboxes'].append([0, 0, image.shape[1], hr_locs[i]-1])
                seperators['horizontal_lines'].append((0, hr_locs[i]-1, image.shape[1], hr_locs[i]-1))
    
    # print(hr_locs)
    
    # image_transpose = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # print(image_transpose.shape)
    # print(image_transpose.shape)
    # cv2.imwrite('img_tran.png',image_transpose)
    # # print(image_transpose[0:image_transpose.shape[0], 0:int(image_transpose.shape[1]*0.90)].shape)
    # # image_transpose = image_transpose[0:image_transpose.shape[0], 0:int(image_transpose.shape[1]*0.90)]
                
    # for loc, vr_px in enumerate(image_transpose):
    #     line_color = get_line_color(vr_px, pixel_count = image.shape[0], threshold=99.8)
    #     if line_color == 1:
    #         seperators['vertical_lines'].append((loc, 0, loc, image.shape[0]))
            

    
    # for loc, vr_px in enumerate(image_transpose):
    #     line_color = get_line_color(vr_px, pixel_count = image.shape[0], threshold=65)
    #     if line_color == 0:
    #         seperators['vertical_lines'].append((loc, 0, loc, image.shape[0]))

    
    # for i in range(len(vr_locs)-2):
    #     # if vr_locs[i] + 3 < vr_locs[i+1]:
    #         # if np.count_nonzero(image_transpose[vr_locs[i]]) > 0:
    #         #     new_loc = get_non_zero_loc(image_transpose, vr_locs[i], threshold=0.9)
    #             seperators['vertical_lines'].append((new_loc, 0, new_loc, image.shape[0]))
            # else:
            #     seperators['vertical_lines'].append((vr_locs[i]-1, 0, vr_locs[i]-1, image.shape[0]))
        
    

    # for loc, vr_px in enumerate(image):
    #     line_color = get_line_color(vr_px, pixel_count = image.shape[0])
    #     if line_color == 0:
    #         # print(loc, seperators['vertical_lines'])
    #         # if len(seperators['vertical_lines']) == 0:
    #         #     seperators['vertical_lines'].append((loc, 0, loc, image.shape[0]))
    #         # elif loc - seperators['vertical_lines'][len(seperators['vertical_lines'])-1][0] > 5:
    #         # elif loc - seperators['vertical_lines'][len(seperators['vertical_lines'])-1][0] <= 20:
    #             seperators['vertical_lines'].append((loc, 0, loc, image.shape[0]))
            
    
    return seperators, image


def remove_all_black(image, threshold=0.1):
    for i in range(image.shape[0]):
        if np.count_nonzero(image[i])/image[1].size <= threshold:
            image[i] = 255
    return image

def get_non_zero_loc(image, loc, threshold=0.99):
    for i in range(loc-1, 0, -1):
        if np.count_nonzero(image[i])/image[i].size >= threshold:
            return i
    
    return loc

def get_line_color(pixel_strip, pixel_count, threshold=90, return_thresold=False):
    white_pxs = np.count_nonzero(pixel_strip)
    white_px_qty = (white_pxs / pixel_count)*100
    if white_px_qty > threshold:
        if return_thresold:
            return 1, white_px_qty
        return 1
    if return_thresold:
        return 0, white_px_qty
    return 0

def get_major_color(image_array, k=20, axis=0):
    # print(image_array.shape)
    sub_array = image_array[:k]
    colors = np.zeros(k)
    if axis == 1:
        sub_array = cv2.rotate(sub_array, cv2.ROTATE_90_CLOCKWISE)
    for i, px_strip in enumerate(sub_array):
        white_pxs = np.count_nonzero(px_strip)
        
        white_px_qty = (white_pxs / sub_array.shape[not axis])*100
        if white_px_qty > 90:
            colors[i] = 1
    unique, counts = np.unique(colors, return_counts=True)
    if unique.size == 1:
        return int(unique[0])
    
    if counts[0] > counts[1]:
        return 0
    return 1
    
    

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

def binarize(img):
    otsu_threshold, image_res = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY +  cv2.THRESH_OTSU)
    return otsu_threshold, image_res

def bgs_in_images(image_array, threshold):

    image_bottom = image_array[int(image_array.shape[0]*0.5):image_array.shape[0]]

    black_pixels = np.column_stack(np.where(image_bottom < threshold))
    white_pixels = np.column_stack(np.where(image_bottom > threshold))

    if black_pixels.size < white_pixels.size:
        main_bg = 1
    else:
        main_bg = 0
    header_bg = get_major_color(image_array, k=100, axis=0)
    
    
    print(f"Main BG: {main_bg},\nHeader BG: {header_bg},\nN black px: {black_pixels.size},\nN white px: {white_pixels.size}\n")

    return main_bg, header_bg, white_pixels, black_pixels
    # white_to_black_ratio = white_pixels.size / black_pixels.size
    # print(white_to_black_ratio)
    # grouped_pixels_y = continous_pixels(white_pixels, axis=0)

    # print(grouped_pixels_y)


# images_path = '/home/gayathri/table_processing/data/output/grassim_104/tables'
# images_list = os.listdir(images_path)

# save_to = '/home/gayathri/table_processing/data/output/grassim_rd_test'
# os.makedirs(save_to, exist_ok=True)
# os.makedirs(os.path.join(save_to, 'binarised'), exist_ok=True)
# os.makedirs(os.path.join(save_to, 'res'), exist_ok=True)


# for image_name in images_list:
#     print(image_name)
#     image = cv2.imread(os.path.join(images_path, image_name), 0)
#     print(image.shape)
#     threshold, bin_image = binarize(image)
    
#     main_bg, header_bg, white_pixels, black_pixels = bgs_in_images(bin_image)
#     lines, image_no_lines = get_areas(bin_image, white_pixels=white_pixels, black_pixels=black_pixels, main_bg=main_bg, header_bg=header_bg)
#     color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#     cv2.imwrite(os.path.join(save_to, 'binarised', image_name), image_no_lines)
#     if lines != []:
#         for line in lines['row_bboxes']:
#             cv2.rectangle(color_image, line[:2], line[2:],(0, 244, 0), 2)
#         # for line in lines['vertical_lines']:
#         #     cv2.rectangle(color_image, line[:2], line[2:],(0, 244, 0), 2)
#         cv2.imwrite(os.path.join(save_to, 'res', image_name), color_image)
    # exit()
    # cv2.imwrite('test.png', bin_image)

    # exit()