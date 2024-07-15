"""
Description
-----------

Perform OCR using AWS textract.
Update line no. 251 to update the Images path for which you want to 
apply the OCR operation.
"""


#Analyzes text in a document stored in an S3 bucket. Display polygon box around text and angled text 
import boto3
import json
import os
import copy
from tqdm import tqdm
from PIL import Image, ImageDraw

def ShowBoundingBox(draw,box,width,height,boxColor):
             
    left = width * box['Left']
    top = height * box['Top'] 
    draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline=boxColor)   
    return [left,top, left + (width * box['Width']), top +(height * box['Height'])]

def process_text_analysis(client, document, image_name):

    
    # # Get the document from S3
    # try:                          
    #     with open(document, 'rb') as f:
    #     stream = f.read()
    # except Exception as e:
    #     print(f'The')
    try:
        with open(document,'rb') as f:
            stream= f.read()
    except Exception as e:
        print(e)
        print(f'The file {image_name} is not able to open in binary!')

    image_org=Image.open(document)
    image = Image.new('RGBA', image_org.size)
    image.paste(image_org)

    # Analyze the document
    image_binary = stream #.getvalue()
    try:
        # print(type(image_binary))
        # print(f'the image name in function: {document}')
        response = client.analyze_document(Document={'Bytes': image_binary},
            FeatureTypes=[])
    except Exception as e:
        print(e)
        print(f'The file {image_name} is not able run by the service')
    
    #Get the text blocks
    blocks=response['Blocks']
    width, height =image.size    

    words_map = {}

    # metadata = []
    # Create image showing bounding box/polygon the detected lines/text
    for i, block in enumerate(blocks):
        draw=ImageDraw.Draw(image)

        if block['BlockType'] == "WORD":
            coords = ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height,'white')
            coords = [int(item) for item in coords]
            words_map.update({block['Id']: {'text':block['Text'], 'bbox': coords}})

    return words_map

def main():

    # session = boto3.Session(profile_name='profile-name')
    # s3_connection = session.resource('s3')
    client = boto3.client('textract', region_name='ap-south-1', aws_access_key_id='AKIA5QCZ7JK43AUK35DF',
                                   aws_secret_access_key='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',endpoint_url='https://textract.ap-south-1.amazonaws.com')

    

    
    img_path = "../annotated/annotated_img"
    # img_path= os.path.join(root_path,'Images')
    images_list = os.listdir(img_path)

    save_to = '../ocr'
    
    os.makedirs(save_to, exist_ok=True)

    for image_name in tqdm(images_list, desc="Performing OCR"):
        # print(f'image name: {image_name}')
        document = os.path.join(img_path, image_name)
        all_words =process_text_analysis(client, document,image_name)
        # print("Blocks detected: " + str(block_count))
        
        json_name = image_name.replace('png', 'json')
        
        with open(os.path.join(save_to, json_name), 'w') as f:
            json.dump(all_words, f, indent=4)

if __name__ == "__main__":
    main()
