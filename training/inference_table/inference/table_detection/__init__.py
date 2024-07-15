import os
import cv2
import numpy as np
from inference.table_detection.Constants import Constants
#import matplotlib.pyplot as plt
import logging

def plot_prediction(img, predictor, strImageFolder, strPageNum, strFileName):
    outputs = predictor(img)

    # Blue color in BGR 
    color = (255, 0, 0) 
  
    # Line thickness of 2 px 
    thickness = 2
    left=0
    top=0
    bottom=0
    right=0
    table_width = 0

    for x1, y1, x2, y2 in outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy():
        start_point = (round(x1), round(y1)) 
  
        end_point = (round(x2), round(y2)) 
   
        if(table_width < round(x2-x1)):
            table_width = round(x2-x1)
            table_height = round(y2-y1)
            right=x2
            left=x1 
            bottom=y2
            top=y1 
        
    img = cv2.rectangle(np.array(img, copy=True), (round(left),round(top)), (round(right),round(bottom)), color, thickness)
    cropped_img = img[round(top):round(bottom), round(left):round(right)]
    # Displaying the image
    try:
        os.unlink(os.path.join(strImageFolder, Constants.strTableDetectedFolder, Constants.strTableDetectedFileSuffix  + Constants.strImageFileFormat))
    except Exception as e:
        logging.error(e)
    if not os.path.isdir(os.path.join(strImageFolder, Constants.strTableDetectedFolder)):
        os.makedirs(os.path.join(strImageFolder, Constants.strTableDetectedFolder))
    if top != bottom and left != right:
        cv2.imwrite(os.path.join(strImageFolder, Constants.strTableDetectedFolder, strFileName  + Constants.strImageFileFormat),img)

    try:
        os.unlink(os.path.join(strImageFolder, Constants.strDetectedTableCropFolder, strFileName + Constants.strImageFileFormat))
    except Exception as e:
        logging.error(e)
    if not os.path.isdir(os.path.join(strImageFolder, Constants.strDetectedTableCropFolder)):
        os.makedirs(os.path.join(strImageFolder, Constants.strDetectedTableCropFolder))

    if top != bottom and left != right:
        cv2.imwrite(os.path.join(strImageFolder, Constants.strDetectedTableCropFolder, strFileName),cropped_img)

    return left,top,right,bottom
    

def make_prediction(img, predictor):
    

    """
    ToDo: Order the res based on confidence scores
    """


    #img = cv2.imread(img_path)
    outputs = predictor(img)
    scores = outputs["instances"].get_fields()["scores"].to("cpu").numpy()
    # table_list = []
    # table_coords = []
    
     #{'coords': [], 'conf':[]}

    table_data = []

    

    if np.any(scores):
        score_index = np.argsort(-scores)
        # score_index = score_index[::-1]
        # max_idx = np.argmax(scores)
        print(score_index)

        for idx in score_index: #range(len(scores)):
            x1, y1, x2, y2 = outputs["instances"].get_fields()["pred_boxes"].tensor.to('cpu').numpy()[idx]
            
            table_data.append({'coords': [int(x1),int(y1),int(x2),int(y2)], 'conf': float(scores[idx])})
            # table_data['table_list'].append(np.array(img[int(y1):int(y2), int(x1):int(x2)], copy=True))
            

    return table_data
