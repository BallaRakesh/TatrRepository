import requests
from base64 import b64encode
from PIL import Image 
import io 
import cv2
 
def rotate_image(encoded_data, headers, url):
    
    parameters = {
            "imgdata" : encoded_data,
            "AdaptiveBinarization" : {
                "enable": False,
                "window_size":3,
                "background_removal": 10,
                "min_threshold": 75,
                "max_threshold":220
            },
            "AutoDetectOrientation" : { 
                "enable": True,
                "mode": 1
            },
            "AutoDeskew" : {
                "enable": False,
                "accuracy":2
            },
            "RemoveTick" : {
                "enable": False,
                "tick_specs": {
                    "MinTickLength": 0.075, 
                    "MaxTickStripHeight": 0.05, 
                    "MinTickAngle": 10, 
                    "MaxTickAngle": 75, 
                    "CharPreservation": 1,
                    "PreferencesFlag": 1
                }
            },
            "MedianFilter" : {
                "enable": False,
                "filter_size": 3,
                "option": 0
            },
            "DespeckleImage" : {
                "enable": False,
                "speckle_threshold": 55
            },
            "AdaptiveBackgroundClean" : {
                "enable": False,
                "window": 1,
                "foreground_enhancement_flag": 1
            },
            "RemoveNoiseEx":{
                "enable": False,
                "erosion_dim": 94,
                "advance": 1,
                "dis_to_remove": 0.075
            },
            "PerformHorizontalSmearing":{
                "enable": False,
                "horizontal_threshold": 10
            },
            "PerformVerticalSmearing":{
                "enable": False,
                "vertical_threshold": 10
            }
        }

    
    response = requests.request("POST", url, headers=headers, json=parameters).json()
    print(response)
    
    exit()
    
    
def correct(encoded_image):
    
    
    headers = {
    'Content-Type': 'application/json'
    }
    
    url = "http://10.2.0.7:5001/process-image"
    
    rotate_image(encoded_image, headers, url)
    
    
image_path = '/New_Volume/master_table_extraction/IM_1197969952.pdf.png'
# image = Image.open(image_path).convert('RGB')
image = cv2.imread(image_path)
bmp_img = cv2.imencode('.bmp', image)
encoded_image = b64encode(bmp_img[1]).decode('utf-8')

# temp = io.BytesIO()
# image.save(temp,  format='bmp')

# with open(temp, 'rb') as f:
#     encoded_image = b64encode(f.read()).decode('utf-8')
    
correct(encoded_image)


# import base64
# import json
# import numpy as np
# import requests
# from base64 import b64encode, b64decode
# import time
# import os
# from PIL import Image
# import io
# import cv2


# headers = {
#   'Content-Type': 'application/json'
# }

# url = "http://10.2.0.7:5001/process-image"

# # image_path = "Test"
# # images = os.listdir(image_path)


# class ImagingAPIs():
#     """
#         The rest api is used to perform the transformations on a given image document.

#         The request will be sent to url: http://10.2.0.7:5001/process-image

#         Method: POST
#     """
#     def __init__(self, cleanInfo = {}):

#         return
    

#     def handle_api_IO(self):

#         # This loop will process each image one by one.
#         # for img in images:
#         image = '/New_Volume/master_table_extraction/IM_1197969952.pdf.png' #os.path.join(image_path, img)
#         print(image)
#         with open(image, 'rb') as f:
#             data = f.read()
        
#         image_2 = Image.open(io.BytesIO(data))
#         image_2 = image_2.convert("RGB")
#         print(image_2.size)
#         image_2.save("img.jpg", 'BMP')
#         print("image_2 =========1====================> ", image_2 )

#         with open("img.jpg", 'rb') as f:
#             data = f.read()
#         image_2 = Image.open(io.BytesIO(data))
#         # image_2.show()
#         print("image_2 ==========2===================> ", image_2 )
#         encoded_data = b64encode(data).decode('utf-8')
#         self.start_end_API_requests(encoded_data)            

#         return
    
#     def start_end_API_requests(self, encoded_data):
#         print(" Starting =============================================================")


#         param = self.prepare_api_parameter(encoded_data)
#         response_json = self.call_api_with_parameters(param)

#         res_img = response_json['output'][1]

#         if isinstance(res_img, str) and len(res_img) > 100:
#             print("Returned Image string")

#             image_bytes = b64decode(res_img)
#             image = Image.open(io.BytesIO(image_bytes))
#             # image.show()
#             # print(image)
#             # os.mkdir("img11.jpg")
#             image = image.save(f"img11_02.bmp", 'BMP')


#             # Dump this image file

#         else:
#             print(response_json)
#         return

#     def prepare_api_parameter(self, encoded_data):
#         print(" ++++++ Inside prepare API ++++++ ")
#         parameters = {
#             "imgdata" : encoded_data,
#             "AdaptiveBinarization" : {
#                 "enable": False,
#                 "window_size":3,
#                 "background_removal": 10,
#                 "min_threshold": 75,
#                 "max_threshold":220
#             },
#             "AutoDetectOrientation" : { 
#                 "enable": True,
#                 "mode": 1
#             },
#             "AutoDeskew" : {
#                 "enable": False,
#                 "accuracy":2
#             },
#             "RemoveTick" : {
#                 "enable": False,
#                 "tick_specs": {
#                     "MinTickLength": 0.075, 
#                     "MaxTickStripHeight": 0.05, 
#                     "MinTickAngle": 10, 
#                     "MaxTickAngle": 75, 
#                     "CharPreservation": 1,
#                     "PreferencesFlag": 1
#                 }
#             },
#             "MedianFilter" : {
#                 "enable": False,
#                 "filter_size": 3,
#                 "option": 0
#             },
#             "DespeckleImage" : {
#                 "enable": False,
#                 "speckle_threshold": 55
#             },
#             "AdaptiveBackgroundClean" : {
#                 "enable": False,
#                 "window": 1,
#                 "foreground_enhancement_flag": 1
#             },
#             "RemoveNoiseEx":{
#                 "enable": False,
#                 "erosion_dim": 94,
#                 "advance": 1,
#                 "dis_to_remove": 0.075
#             },
#             "PerformHorizontalSmearing":{
#                 "enable": False,
#                 "horizontal_threshold": 10
#             },
#             "PerformVerticalSmearing":{
#                 "enable": False,
#                 "vertical_threshold": 10
#             }
#         }

#         return parameters


#     def call_api_with_parameters(self, param):
        
#         # payload = json.dumps(param)
#         response = requests.request("POST", url, headers=headers, json=param)
#         res = response.json()

#         try:
#             print("response type =====try=====> \n", type(res['output']), len(res['output']), res["output"][0])
#         except:
#             print("response type ====except======> \n", res.keys())


#         return res


# if __name__ == "__main__":

#     obj = ImagingAPIs()
#     obj.handle_api_IO()
