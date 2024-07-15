ABBY_API = "https://abbyocr.centralindia.cloudapp.azure.com/FineReaderServer14/api"
ABBY_PROCESSING_ENDPOINT = "workflows/Default%20Workflow/input/ticket"
ABBYY_STATUS_ENDPOINT = "jobs/%7B{}%7D"
ABBY_TEXT_RESULT_ENDPOINT = "result/outputDocuments/1/files/0"
ABBY_JSON_RESULT_ENDPOINT = "result/outputDocuments/2/files/0"
ABBY_HEADER = {'Content-Type': 'application/json'}
LANGUAGE_ABBYY = "languageAbbyy"
INPUT_FILES = "InputFiles"
FILE_CONTENTS = "FileContents"
RECOGNITION_PARAMS = "RecognitionParams"
LANGUAGES = "Languages"
RECOGNITION_QUALITY = "recognitionQuality"
RECOGNITION_QUALITY_PREFFIX="RQS_"
LAYOUT = "layout"
PAGES = "pages"
TEXTS = "texts"
LINES = "lines"
WORDS = "words"
SMALL_CONFIDENCE = "confidence"
POSITION = "position"
JS_COMPLETE = "JS_Complete"
STATE = "State"
JS_NOSUCHJOB = "JS_NoSuchJob"
PROGRESS = "Progress"
FILENAME = "FileName"

import json
import requests
import base64
import io
from tqdm import tqdm
import os
from PIL import Image

import ocr_constants


    
def get_config(image):
    # base64_string = pil_image_to_base64(image)
    image_buffer = io.BytesIO()
    image.save(image_buffer,format="PNG")
    image_buffer.seek(0)
    base64_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    with open("abby_config.json","r") as config:
        config_file = json.load(config)
    config_file[ocr_constants.INPUT_FILES][0]["FileData"][ocr_constants.FILE_CONTENTS] = base64_image
    # config_file[ocr_constants.RECOGNITION_PARAMS][ocr_constants.LANGUAGES] = [language]
    # config_file[ocr_constants.INPUT_FILES][0]["FileData"][ocr_constants.FILENAME] = file_name.split("/")[-1]
    # config_file[ocr_constants.RECOGNITION_PARAMS][ocr_constants.RECOGNITION_QUALITY] = "{}{}".format(ocr_constants.RECOGNITION_QUALITY_PREFFIX,recognition_mode)
    return config_file

def get_word_cordinates(json_result, image_w, image_h):
    word_coordinates = []
    img_character_count = 0
    img_conf = 0
    word_coordinates = []
    for page in json_result[ocr_constants.LAYOUT][ocr_constants.PAGES]:
        width = page['width']
        height = page['height']
        horizontal_resize = width / image_w
        vertical_resize = height / image_h
        for text_block in page[ocr_constants.TEXTS]:
            for line in text_block[ocr_constants.LINES]:
                for word_data in line[ocr_constants.WORDS]:
                    word = word_data[ocr_constants.SMALL_TEXT]
                    confidence = word_data[ocr_constants.SMALL_CONFIDENCE]
                    position = word_data[ocr_constants.POSITION]
                    img_character_count += len(word)
                    img_conf += confidence

                    # width = position['r'] - position['l']
                    # height = position['b'] - position['t']

                    x1 = int(position['l']//horizontal_resize)
                    y1 = int(position['t']//vertical_resize)
                    x2 = int(position['r']//horizontal_resize)
                    y2 = int(position['b']//vertical_resize)
                    width = x2 - x1
                    height = y2 - y1
                    vertices = [(x1, y1), 
                                (x1+width, y1), 
                                (x1+width, y1+height), 
                                (x1, y1+height)]

                    word_coordinates.append({
                        ocr_constants.WORD: word,
                        ocr_constants.VERTICES: vertices
                    })

                    # word_coordinates.append(word_info)

        
        for table in page['tables']:
                for cells in table['cells']:
                    # print(cells.keys())
                    for lines in cells['lines']:
                        for word in lines['words']:
                            # if word_coordinates != {}:
                            #     word_id = max(word_coordinates.keys()) + 1
                            # print(word, end='\n\n')
                            x1 = int(word['position']['l']//horizontal_resize)
                            y1 = int(word['position']['t']//vertical_resize)
                            x2 = int(word['position']['r']//horizontal_resize)
                            y2 = int(word['position']['b']//vertical_resize)
                            width = x2 - x1
                            height = y2 - y1
                            vertices = [(x1, y1), 
                                        (x1+width, y1), 
                                        (x1+width, y1+height), 
                                        (x1, y1+height)]
                            word_coordinates.append({
                                                        "text": word['text'],
                                                        "vertices": vertices
                                                    })
                                
    return {i:word_coordinates[i] for i in range(len(word_coordinates))}

def generate_ocr_string_and_word_coordinates(image):
    config_file = get_config(image)
    main_response = requests.post("{}/{}".format(ABBY_API,ABBY_PROCESSING_ENDPOINT), headers=ABBY_HEADER,
                                    data=json.dumps(config_file),
                                    verify=False)
    try:
        job_id = main_response.text.replace("{", "").replace("}", "").replace('"','')
        print("job id is .......",job_id)
    except:
        raise Exception(main_response.text)
    
    api_status_and_result_endpoint = "{}/{}".format(ABBY_API,ABBYY_STATUS_ENDPOINT.format(job_id))
    while True:
        status_response = requests.get(api_status_and_result_endpoint, verify=False)
        stateInfo = status_response.json()
        if stateInfo[ocr_constants.STATE] == ocr_constants.JS_COMPLETE:
            # print("  Job complete")
            break
        elif stateInfo[ocr_constants.STATE] == ocr_constants.JS_NOSUCHJOB:
            raise Exception("Job \"%s\" not found" % (job_id))
        # else:
            # print("  Job state is %s, %d%% complete" % (stateInfo[ocr_constants.STATE], stateInfo[ocr_constants.PROGRESS]))

    text_result = requests.get("{}/{}".format(api_status_and_result_endpoint,ABBY_TEXT_RESULT_ENDPOINT), verify=False)
    all_text = text_result.text

    json_result = requests.get("{}/{}".format(api_status_and_result_endpoint,ABBY_JSON_RESULT_ENDPOINT) , verify=False)
    try:
        json_result = json_result.json()
    except:
        json_result = {}

    if json_result != {}:
        image_w, image_h = image.size
        word_coordinates = get_word_cordinates(json_result, image_w, image_h)
    else:
        all_text = ''
        word_coordinates = {}

    return all_text, word_coordinates

    

if __name__ == '__main__':
    image_path = '/home/gayathri/table_processing/data/input/grasim_104/deskewed'
    save_to = '/home/gayathri/table_processing/data/input/grasim_104/ocr'
    os.makedirs(save_to, exist_ok=True)
    image_list = os.listdir(image_path)

    for img in tqdm(image_list, 'Processing'):
        jsonname = img.replace('png','json')
        if not os.path.exists(os.path.join(save_to, jsonname)):
            image = Image.open(os.path.join(image_path, img))

            text, coords = generate_ocr_string_and_word_coordinates(image=image)

            with open(os.path.join(save_to, jsonname), 'w') as f:
                json.dump(coords, f, indent=4)

        # image_array = np.asarray(image)
        # for word in coords:
            
        #     cv2.rectangle(image_array, coords[word]['vertices'][0], coords[word]['vertices'][2], (0, 244, 0), 2)
        # print(coords)
        # cv2.imwrite('test_abby_ocr.png', image_array)
        # exit()