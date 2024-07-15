#Change History:
#Date            Author              
#06/11/2023      Utkarsh Srivastava

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

import sys
sys.path.append('../')
import utils

import logging


# def correct_annotations(annotations, image_props):
#     keep_annotations = {}
#     # #print(image_props.header_data)
#     x_change = image_props.header_data[0]
#     y_change = image_props.header_data[1]
    
#     # #print(annotations)
#     # #print()
#     for word in annotations:
        
#         iou = utils.calculate_iou(annotations[word]['bbox'], image_props.header_data)
#         if iou > 0:
#             # #print(annotations[word])
#             for i in range(len(annotations[word]['vertices'])):
#                 annotations[word]['vertices'][i][0] -= x_change
#                 annotations[word]['vertices'][i][1] -= y_change
#             annotations[word]['left'] -= x_change
#             annotations[word]['top'] -= y_change
#             annotations[word]['x1'] -= x_change
#             annotations[word]['x2'] -= x_change
#             annotations[word]['y1'] -= y_change
#             annotations[word]['y2'] -= y_change
#             annotations[word]['bbox'] = [annotations[word]['x1'], annotations[word]['y1'], 
#                                          annotations[word]['x2'], annotations[word]['y2']]
#             keep_annotations[word] = annotations[word]
#             # #print(annotations[word])
#     # #print(keep_annotations)
#     return keep_annotations

def get_image(image_path, img_props, crop_trash=False):
    return cv2.imread(image_path) if crop_trash != True else img_props.crop_image()

def line_preprocess(original_image,annotations):

    line_coordinates = []
    column_line_coordinates = []

    processed_image = original_image.copy()
    lines_image = original_image.copy()
    height, width = original_image.shape[:2]
    cnt=0
    # #print(annotations)
    
    if annotations:
        
        loop_start = time.time()
        
        for word in annotations:
            cnt+=1
            vertices =  annotations[word]['vertices'] #[vertex for vertex in annotation.bounding_poly.vertices]
            # vertices = [(vertex.x, vertex.y) for vertex in vertices]
       
            vertices = np.array(vertices, dtype=np.int32)
            # try:
            x= vertices[0][0]#+1
            y= vertices[0][1]#+1
            pixel_color = processed_image[y, x]
            # #print(pixel_color)
            # try:
            blue, green, red = pixel_color
            # except:
            #     blue, green, red = 255, 255, 255
            # if cnt>=2:
            # if vertices[0][0]+1 < processed_image.shape[1]:
            #     x= vertices[0][0]+1
            # else:
            #     x = processed_image.shape[1]-2
            
            # if vertices[0][1]-1 < processed_image.shape[0]:

            #     y= vertices[0][1]-1
            # else:
            #     y = processed_image.shape[0] - 2
            

           
            # except :
            #     blue, green, red = [255, 255, 255]
            cv2.fillPoly(processed_image,[vertices],color = (int(blue), int(green), int(red)))
            # cv2.imwrite('check.png', processed_image)
            image = processed_image
            gray = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            use, thres = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

            for i in contours:
                cnt = cv2.contourArea(i)
                if cnt < 500:
                    x,y,w,h = cv2.boundingRect(i)
                    cv2.rectangle(thres,(x-1,y-1),(x+w+1,y+h+1),(255,255,255),-1)
            # cv2.imwrite('thres.png', thres)
            edges = cv2.Canny(thres, 50, 150, apertureSize=3)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.erode(edges, kernel)
            # cv2.imwrite('edges.png', edges)
            edge_coordinates = np.argwhere(edges > 0)  # Returns a list of (x, y) coordinates
        number_of_pixel = height
        # edge_coordinates = utils.sort_coord(edge_coordinates)
        edge_coordinates = sorted(edge_coordinates, key=lambda x: x[1])
        # #print("MAX OF EDGE COORDS",max([x[1] for x in edge_coordinates]))
        width_arr = [0]*width
        
        # #print(f"WIDTH_ARR: {len(width_arr)}")
        arr = []
        if edge_coordinates != []:
            i = edge_coordinates[0][1]
            flag = 0
            for x, y in edge_coordinates:
                if flag == 1:
                    i = y
                    flag = 0
                    arr = []
                if y<=i+2:
                    arr.append(x)
                else:
                    pix  = len(set(arr))
                    pix = len(set(arr))
                    # #print(f"I: {i}")
                    width_arr[i-1] = pix
                    flag = 1

            for x1 in range(len(width_arr)):
                profile = width_arr[x1]
                if ( profile / (number_of_pixel))*100 >=60:
                    start_point = (x1+2,0)
                    end_point = (x1+2,height)
                    line_coordinates.append([start_point,end_point])
        if line_coordinates != []:
            column_line_coordinates.append(line_coordinates[0])
            for i in range(0,len(line_coordinates)-1):
                if abs(line_coordinates[i][0][0]-line_coordinates[i+1][0][0]) >= 4:
                    column_line_coordinates.append(line_coordinates[i+1]) 
        # #print(f"After second loop: {time.time() - loop2_time}")
    return column_line_coordinates, lines_image



def column_preprocess(original_image,annotations):
    
    cnt=0
    image_path = original_image.copy()
    if annotations:
        # #print(annotations)s
        processed_image = np.ones_like(original_image) * 255
        
        contours = []   
        use_contours = [] 
        for word in annotations:
            cnt+=1
            vertices = annotations[word]['vertices'] #[vertex for vertex in annotation.bounding_poly.vertices]
            
            # vertices = [(vertex.x, vertex.y) for vertex in vertices]
            vertices_new = list(vertices)
            use_contours.append(vertices_new)

            # #print(vertices_new)
            x1 = vertices[0][0]
            y1 = vertices[0][1]
            
            xw = vertices[1][0]
            yh = vertices[2][1]
            vertices_new = [(x1-2, y1-2), (xw+2, y1-2), (xw+2, yh+2), (x1-2, yh+2)]
        
            contours.append(vertices)
            
        
            vertices = np.array(vertices_new, dtype=np.int32)

            # if cnt>1:
            cv2.polylines(processed_image, [vertices], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.fillPoly(processed_image,[vertices],0, 255, 0)

        contours.sort(key=lambda x: x[0][1])
        merged_contours = []
        current_row = []
        for contour in contours:
            if not current_row:
                current_row.append(contour)
            else:
                # Check if the current contour is in the same row as the previous one
                if abs(current_row[-1][0][1] - contour[0][1]) < 20:  # Adjust the threshold as needed
                    current_row.append(contour)
                else:
                    merged_contours.append(current_row)
                    current_row = [contour]


        merged_img = original_image.copy()

        for merged_row in merged_contours:
            if len(merged_row) > 1:
                x1 = min(merged_row, key=lambda x: x[0][0])[0][0]
                y1 = min(merged_row, key=lambda x: x[0][1])[0][1]
                x2 = max(merged_row, key=lambda x: x[1][0])[1][0]
                y2 = max(merged_row, key=lambda x: x[1][1])[1][1]
                cv2.rectangle(merged_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        image = copy.deepcopy(processed_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        white_pixel_threshold = 255

        horizontal_projection = np.sum(image, axis=0)

        height, width = image.shape[:2]

        image_with_lines = image.copy()

        arr=[]
        contours = []
        for row, projection_value in enumerate(horizontal_projection):
            if projection_value >= white_pixel_threshold * height:
                
                cv2.line(image_with_lines, (row, 0), (row,width - 1), color = (146, 252, 79), thickness=1)
                arr.append(row)

        try:
            arr.sort()
            start=arr[0]
            end = arr[0]
        
            cntr = 1
            for x in range(1,len(arr)):    
                
                if arr[x-1]+1==arr[x]:
                    end=arr[x]
                    cntr+=1
                    

                    
                else:
                    
                    if cntr==0:
                        
                        cntr = 0
                        cv2.rectangle(image_path, (int(start),int(0)), (int(start),int(height-1)), color = (0, 0, 135), thickness=1)
                        contours.append([(int(start),int(0)),(int(start),int(0)),(int(end),int(height-1)),(int(start),int(height-1))])
                        
                        start=arr[x]
                        

                    else:
                        cntr = 0

                        cv2.rectangle(image_path, (int(start),int(0)), (int(end),int(height-1)), color = (0, 0, 135), thickness=cv2.FILLED)
                        
                        contours.append([(int(start),int(0)),(int(end),int(0)),(int(end),int(height-1)),(int(start),int(height-1))])
                        start=arr[x]

                if x==len(arr)-1:
                    if cntr==0:
                        cntr = 0
                        cv2.rectangle(image_path, (int(start),int(0)), (int(start),int(height-1)), color = (0, 0, 135), thickness=1)
                        contours.append([(int(start),int(0)),(int(start),int(0)),(int(end),int(height-1)),(int(start),int(height-1))])
                        start=arr[x]


                    else:
                        cntr = 0
                        
                        cv2.rectangle(image_path, (int(start),int(0)), (int(end),int(height-1)), color = (0, 0, 135), thickness=cv2.FILLED)
                        
                        contours.append([(int(start),int(0)),(int(end),int(0)),(int(end),int(height-1)),(int(start),int(height-1))])
                        start=arr[x]
        except:
            logging.warning("no column detected")

        column_image = image_path
        column_contours = contours
        return column_image,column_contours
    else:
        logging.warning("empty document or OCR failed")
        return None, None

def column_combinator(original_image, annotations, column_coordinate,column_line_coordinates):
    height, width = original_image.shape[:2]
    index = []
    final_column_coordinates  = []
    final_column_line = []
    final_column_coordinate = []
    for j,ele in enumerate(column_coordinate):
        for i in column_line_coordinates:
            if  ele[0][0] - 6 <= i[0][0] <= ele[1][0] + 6:
                index.append(j)
                break

    for i in range(len(column_coordinate)):
        if i not in index:
            final_column_coordinates.append(column_coordinate[i])

    for coord in final_column_coordinates:
        start = (int((coord[0][0]+coord[1][0])/2),coord[0][1])
        end = (int((coord[0][0]+coord[1][0])/2),coord[2][1])
        final_column_line.append([start,end])
        final_column_coordinate.append([start,end])


    cnt=0
    if annotations:

        contours = []   
        use_contours = [] 
        for annotation in annotations:
            cnt+=1
            vertices = annotations[annotation]['vertices']
            vertices_new = list(vertices)
            use_contours.append(vertices_new)


            xr=vertices_new[1][0]
            yr=vertices_new[1][1]
            
            x1=vertices_new[2][0]
            y1=vertices_new[2][1]
            x2=vertices_new[3][0]
            y2=vertices_new[3][1]

            vertices_new[1] = (xr+4,yr)
            vertices_new[2] = (x1+4,y1+4)
            vertices_new[3] = (x2,y2+4)

            contours.append(vertices)


        flag = 0
        if len(final_column_line)>=2  :
            for i in range(len(final_column_line)-1):
                xn1 = final_column_line[i][0][0]
                xn2 = final_column_line[i+1][0][0]
                value = final_column_line[i][0][0]
                for j in contours:
                    x1 = j[0][0]
                    x2  =j[1][0]
                    if (x1<xn1 and x2<xn1) or (x1>xn2 and x2>xn2) :
                        continue
                    else:
                        flag = 1
                        break
                if flag == 0:
                    final_column_coordinate.remove(value)
 

    return final_column_coordinate

# # def get_text_detect(ocr_path, imgprops):
# #     ocr_name = imgprops.image_path.split('/')[-1].replace('.png','.json')
# #     ocr_path = os.path.join(ocr_path,ocr_name)

# #     image_name = imgprops.image_path.split('/')[-1]
    
# #     image = cv2.imread(imgprops.image_path)
# #     header_strip = imgprops.header_data
    
# #     if os.path.exists(ocr_path):
# #         with open(ocr_path, 'r') as f:
# #             ocr_words =json.load(f)
# #     else:
# #         return None, None
# #     table_words = {}
# #     y_start = 0
# #     if header_strip!= None:
# #         y_start = header_strip[1]
# #     for item in ocr_words:
# #         if ocr_words[item]['y1'] >= y_start:
# #             table_words[item] = ocr_words[item]
# #     for key in table_words:
# #         if 'vertices' not in table_words[key].keys():
# #             table_words[key]['vertices'] = [[table_words[key]['x1'], table_words[key]['y1']], 
# #                                             [table_words[key]['x1']+table_words[key]['width'], table_words[key]['y1']], 
# #                                             [table_words[key]['x1']+table_words[key]['width'], table_words[key]['y1']+table_words[key]['height']], 
# #                                             [table_words[key]['x1'], table_words[key]['y1']+table_words[key]['height']]]
# #     return table_words, image
        
def detect_column(annotations, image):
    column_line_coordinates = None
    final_column_coordinate = None
    
    if not annotations == None:
        original_image = copy.deepcopy(image)
        column_line_coordinates, line_image  = line_preprocess(image,annotations)
            
        column_image, column_coordinate = column_preprocess(original_image,annotations)
            
        final_column_coordinate = []
        if column_coordinate!= None:
            final_column_coordinate = column_combinator(image, annotations, column_coordinate,column_line_coordinates)
            for line in final_column_coordinate:
                for pts in range(len(line)):
                    line[pts] = list(line[pts])
                    line[pts] = [int(i) for i in line[pts]]
            final_column_coordinate = [final_column_coordinate]
        for line in column_line_coordinates:
            for pts in range(len(line)):
                line[pts] = list(line[pts])
                line[pts] = [int(i) for i in line[pts]]

        line_coords = {
            'solid_lines':column_line_coordinates,
            'padded_lines':final_column_coordinate
            }
        return line_coords