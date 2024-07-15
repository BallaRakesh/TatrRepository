import base64
import json
import numpy as np
import requests
from base64 import b64encode, b64decode
import time
import os
from PIL import Image
import io
import cv2
import tqdm


headers = {
  'Content-Type': 'application/json'
}

url = "http://10.2.0.7:5001/process-image"

image_path = r"/home/gayathri/table_processing/data/input/grasim_136/images"
images = os.listdir(image_path)

save_loc = r"/home/gayathri/table_processing/data/input/grasim_136/deskewed"
os.makedirs(save_loc, exist_ok=True)

parameters = {
            "imgdata" : None,
            "AutoDeskew" : {
                "enable": True,
                "accuracy":3
            },
            "AutoDetectOrientation":{
                "enable": True,
                "lmode": 1
            }
        }


images_list = os.listdir(image_path)
for img in tqdm.tqdm(images_list, 'Processing'):
    image_name = img.replace('tif','png')
    with open(os.path.join(image_path, img), 'rb') as f:
        image_bin = f.read()
    encoded_data = b64encode(image_bin).decode('utf-8')
    parameters["imgdata"] = encoded_data

    response = requests.request("POST", url, headers=headers, json=parameters)
    res = response.json()

    res_img = res['output'][1]

    image_bytes = b64decode(res_img)
    image = Image.open(io.BytesIO(image_bytes))

    print(os.path.join(save_loc, image_name))
    image.save(os.path.join(save_loc, image_name))