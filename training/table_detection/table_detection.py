import os
import cv2
import numpy as np
from Constants import Constants
#import matplotlib.pyplot as plt

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
        print('')
    if not os.path.isdir(os.path.join(strImageFolder, Constants.strTableDetectedFolder)):
        os.makedirs(os.path.join(strImageFolder, Constants.strTableDetectedFolder))
    if top != bottom and left != right:
        cv2.imwrite(os.path.join(strImageFolder, Constants.strTableDetectedFolder, strFileName  + Constants.strImageFileFormat),img)


    try:
        os.unlink(os.path.join(strImageFolder, Constants.strDetectedTableCropFolder, strFileName + Constants.strImageFileFormat))
    except Exception as e:
        print('')
    if not os.path.isdir(os.path.join(strImageFolder, Constants.strDetectedTableCropFolder)):
        os.makedirs(os.path.join(strImageFolder, Constants.strDetectedTableCropFolder))

    if top != bottom and left != right:
        cv2.imwrite(os.path.join(strImageFolder, Constants.strDetectedTableCropFolder, strFileName),cropped_img)

    # with open('/datadrive/ingramImages/tdtsmvaluation/detect.txt', 'a+') as f:
    #     f.write(strFileName + "," + str(left) + "," + str(top) + "," + str(right) + "," + str(bottom) + '\n')
    return left,top,right,bottom
def make_prediction(img, predictor):
    
    #img = cv2.imread(img_path)
    outputs = predictor(img)
    print(outputs)
    table_list = []
    table_coords = []

    for i, box in enumerate(outputs["instances"].get_fields()["pred_boxes"].tensor.numpy()):
        x1, y1, x2, y2 = box
        x1 = x1-5
        y1=y1-5
        x2=x2+5
        y2=y2+5
        table_list.append(np.array(img[round(y1):round(y2), round(x1):round(x2)], copy=True))
        table_coords.append([round(x1),round(y1),round(x2-x1),round(y2-y1)])
        print("TABLE", i, ":")
        """ cv2.imshow("detect", img[round(y1):round(y2), round(x1):round(x2)])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print() """

    return table_list, table_coords
