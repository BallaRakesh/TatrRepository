from google.cloud import vision
import os
from base64 import b64encode
import json
from tqdm import tqdm

vision_api_key_path = "digital-hall-399509.json"

def get_ocr_vision_api(image_path):
    #ctxt = img_str
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = vision_api_key_path
    with open(image_path, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content = ctxt)

    response = client.text_detection(image=image)

    word_coordinates = {}
    all_text = ""

    for i,text in enumerate(response.text_annotations):
        if i != 0:
            # print('=' * 30)
            # print(text.description)
            vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
            x1 = min([v.x for v in text.bounding_poly.vertices])
            x2 = max([v.x for v in text.bounding_poly.vertices])
            y1 = min([v.y for v in text.bounding_poly.vertices])
            y2 = max([v.y for v in text.bounding_poly.vertices])
            # print('bounds: ' + str(vertices))
            if x2 - x1 == 0:
                x2 += 1
            if y2 - y1 == 0:
                y2 += 1
            
            #print("Confidence: ", text.description, text.confidence)
            word_coordinates[i] = {
                "text": text.description,
                "score" : text.confidence, 
                "left": x1,
                "top": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                'bbox': [x1, y1, x2, y2]
            }
        else:
            all_text = text.description

    return word_coordinates, all_text




image_path = '../benchmark/cropped_no_pad/images'
images = os.listdir(image_path)

os.makedirs('../ocr_no_pad', exist_ok=True)

for enum, image in tqdm(enumerate(images), desc="Processing"):
    # print(image)
    coords, text = get_ocr_vision_api(os.path.join(image_path, image))

    image_name = image.replace('.png', '')
    
    with open(f"../ocr_no_pad/{image_name}.json", 'w') as f:
         json.dump(coords, f, indent=4)