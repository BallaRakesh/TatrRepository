import boto3
import json
import os
from tqdm import tqdm
from PIL import Image
def get_num_pixels(file):
    width, height = Image.open(file).size
    return width,height
def extract_word_coordinates(document_path):
    # Extract text and word-level bounding boxes from the document
    textract = boto3.client('textract', region_name='ap-south-1', aws_access_key_id='AKIA5QCZ7JK43AUK35DF',
                                   aws_secret_access_key='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',endpoint_url='https://textract.ap-south-1.amazonaws.com')
    
  
    response = textract.detect_document_text(Document={'Bytes': open(document_path, 'rb').read()})

    word_coordinates = []
    w,h=get_num_pixels(document_path)
    
    # Extract word coordinates from the response
    for item in response['Blocks']:
        #if not item['BlockType'] == 'WORD':
        print(item)

        if item['BlockType'] == 'WORD':
            word = item['Text']
            x1 = item['Geometry']['BoundingBox']['Left']
            y1 = item['Geometry']['BoundingBox']['Top']
            width = item['Geometry']['BoundingBox']['Width']
            height = item['Geometry']['BoundingBox']['Height']
            
            word_coordinates.append({
                'word': word,
                'left': x1,
                'top': y1,
                'width': width,
                'height': height,
                'x1': x1*w,
                'y1': y1*h,
                'x2': (x1+width)*w,
                'y2': (y1+height)*h
            })
    exit()        
    #print(word_coordinates)
    return word_coordinates

# Path to your document
#document_path = '/home/lpt5355/Downloads/GV/Eswar_Images/IM-000000010652231-AP1.png'

# Extract word coordinates
#coordinates = extract_word_coordinates(document_path)

# Save coordinates to a JSON file
#json_path = '/home/lpt5355/Downloads/GV/Eswar_Images/IM-000000010652231-AP1.png'
#with open(json_path, 'w') as json_file:
 #   json.dump(coordinates, json_file)
image_path = 'dataset/funsd_geo/testing_data/images'
images = os.listdir(image_path)

os.makedirs('textract_table_ocr', exist_ok=True)

for enum, image in tqdm(enumerate(images), desc="Processing"):
    print(image)
    coords = extract_word_coordinates(os.path.join(image_path, image))

    image_name = image.replace('.png', '')
    
    # with open(f"textract_table_ocr/{image_name}.json", 'w') as f:
    #      json.dump(coords, f, indent=4)