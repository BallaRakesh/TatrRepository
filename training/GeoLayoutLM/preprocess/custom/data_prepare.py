"""
Description: 

Preprocessing files to generate data as required by the 
GeoLayout model.

Following information is stored in json files corresponding
to each image

out_json_obj = {
    'blocks':{'first_token_idx_list': [],    # id of first token of each block
              'boxes': []},  # bbox of block
    'words' : [   # info regarding all words
        {
            "text": "",
            "tokens": [],
            "boundingBox": []
    ],  
    'parse' : {
       'class' : {
            class_name : [],   # list all tokens correspondnig to tokens
       }, 
       'relations' : []  # relation between each text segment (whenever key-value pair information is available)
    }
    'meta' : {
       'image_path' : str,
       'imageSize': {"width": int, "height": int},
        'voca' : "bert-base-uncased"
    }
 }

"""

import json
import os
from glob import glob

from PIL import Image

from math import ceil

import shutil

import imagesize
from tqdm import tqdm
from transformers import BertTokenizer
from util import Traintestsplit
from util import DataSegmentation

from itertools import islice

def chunks(data, SIZE=1):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}


def denormalize(h, w, bbox, denom=2):
    """
    Get entire label coordinate region
    
    Parameters
    ----------
    h: int
       heigth of the image
    w: int
       width of the image
    bbox: list
        word coordinates obtained from ocr
    denom: int
        Denominator for denormalize operation

    Returns:
    x0, y0, x1, y1: tuple
        tuple of denormalized word coordinates
    """

    x_center = float(bbox[0]) * w
    y_center = float(bbox[1]) * h
    width = float(bbox[2]) * w
    height = int(float(bbox[3]) * h)
    x0 = int(x_center - (width / denom))
    x1 = int(x_center + (width / denom))
    y0 = int(y_center - (height / denom))
    y1 = int(y_center + (height / denom))

    return x0, y0, x1, y1

def calculate_iou(bbox1, bbox2):
    """
    Finds the percentage of intersection  with a smaller box. (what percernt of smaller box is in larger box)
    """
    # assert bbox1['x1'] < bbox1['x2']
    # assert bbox1['y1'] < bbox1['y2']
    # assert bbox2['x1'] < bbox2['x2']
    # assert bbox2['y1'] < bbox2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    # min_area = min(bbox1_area,bbox2_area)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    intersection_percent = intersection_area / bbox2_area

    return intersection_percent

def get_block_words(block, doc=None):
    block_words = []
    # print(block)
    for words in doc:
        words_iou = calculate_iou(block['bbox'], doc[words]['bbox'])
        if words_iou > 0.4:
            block_words.append(doc[words])
    return block_words
        
def get_class(ocr_bbox, annotations, classes):
    # print(annotations)
    for i in range(len(annotations)):
        iou = calculate_iou(annotations[i], ocr_bbox)
        if iou > 0:
            return CLASSES_VALID[classes[i]]
    return 'O'

def get_links(doc: dict,
              key_val_sets:list):
    
    doc_list = [doc[item] for item in doc]

    for i, token in enumerate(doc_list):
        if not token == '':
            for key_val in key_val_sets:
                key_iou = calculate_iou(token['bbox'], key_val['key_bbox'])
                val_iou = calculate_iou(token['bbox'], key_val['value_bbox']) 
                if key_iou > 0:
                    # print(key_val['key_text'], token['text'])
                    # print(key_val['key_bbox'], token['bbox'])
                    if 'key_token_idx' not in key_val:
                        key_val['key_token_idx'] = [token['token_idx']]
                    else:
                        key_val['key_token_idx'].append(token['token_idx'])
                    doc_list[i] = ''
 
                elif val_iou > 0:
                    if 'value_token_idx' not in key_val:
                        key_val['value_token_idx'] = [token['token_idx']]
                    else:
                        key_val['value_token_idx'].append(token['token_idx'])
                    
                    doc_list[i] = ''
    return key_val_sets
                    
            
    # for item in key_val_sets:
    #     key_xy_min = item['key_bbox']
    #     print(key_xy_min)
    # for i, item in enumerate(all_words_tokenized_list):
    #     print(all_words_bbox_xy_min[i])
    exit()
    # for item in key_val_set:
    #     # print(item, end='\n\n')
    #     if item['key_text_bbox'][0] in all_words_tokenized:
    #         pass

def get_blocks(all_blocks: dict, 
               doc: dict, 
               keep_blocks: list = ['LINE', 'KEY_VALUE_SET' ,'TABLE']):
    """
    To get a better understanding of blocks read - 
    https://github.com/AlibabaResearch/AdvancedLiterateMachinery/issues/36
    """
    blocks_data = {}
    for i, block in enumerate(all_blocks):
        
        if all_blocks[block]['block_type'] in keep_blocks:

            blocks_data[i] = {}
            blocks_words = get_block_words(all_blocks[block], doc)
            if len(blocks_words) >= 1:
                blocks_data[i]['token_data'] = blocks_words
                blocks_data[i]['block_data'] = {'block_type': all_blocks[block]['block_type'],
                                            'block_bbox': [int(item) for item in all_blocks[block]['bbox']]}
    return blocks_data        

# def 


MAX_SEQ_LENGTH = 512
MODEL_TYPE = "bert"
VOCA = "bert-base-uncased"

classes_path = "/home/administrator/Downloads/covering_schedule_complete_data/label.txt"
with open(classes_path, 'r') as f:
    classes = f.readlines()

CLASSES = [item.replace('\n', '').strip() for item in classes]
CLASSES.insert(0, 'O') #, "HEADER", "QUESTION", "ANSWER"]

keep_blocks = ['LINE', 'TABLE']

CLASSES_VALID = CLASSES[1:] 

INPUT_PATH = "/home/administrator/Downloads/covering_schedule_complete_data/"
anno_dir = 'annotations'


OUTPUT_PATH = os.path.join(INPUT_PATH, 'dataset_2')
ANNOTATION_SAVE_PATH = os.path.join(OUTPUT_PATH, "preprocessed/")
os.makedirs(ANNOTATION_SAVE_PATH, exist_ok=True)

train_save_path = os.path.join(OUTPUT_PATH, 'Train')
test_save_path = os.path.join(OUTPUT_PATH, 'Test')

os.makedirs(train_save_path, exist_ok=True)
os.makedirs(test_save_path, exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_PATH, "preprocessed"), exist_ok=True)

print(OUTPUT_PATH)

annotations_path = os.path.join(INPUT_PATH, 'Labels')

images_path = os.path.join(INPUT_PATH, 'Images')
images_list = os.listdir(images_path)

all_words_path = os.path.join(INPUT_PATH, 'ocr/all_words')
key_val_path = os.path.join(INPUT_PATH, 'ocr/key_val_sets')
blocks_path = os.path.join(INPUT_PATH, 'ocr/blocks')

########## Split train and test data ##############

test_size = 0.8
train_idx = ceil(len(images_list) * 0.8)
train_set = images_list[:train_idx]
test_set = images_list[train_idx:]

# train_folder = os.path.join(INPUT_PATH, 'Train')
# val_folder = os.path.join(INPUT_PATH,'Test')
# Traintestsplit(INPUT_PATH, train_folder, val_folder)
###################################################

tokenizer = BertTokenizer.from_pretrained(VOCA, do_lower_case=True)

save_test = os.path.join(OUTPUT_PATH, 'preprocessed_files_val.txt')
save_train = os.path.join(OUTPUT_PATH, 'preprocessed_files_train.txt')

if os.path.exists(save_test):
    os.remove(save_test)
if os.path.exists(save_train):
    os.remove(save_train)

save_test_name = open(save_test, 'a+')
save_train_name = open(save_train, 'a+')

for image_name in tqdm(images_list, 'Processing'):
    
    # image_path = os.path.join(images_path, image_name)

    json_file_name = image_name.replace('png', 'json')
    txt_file_name = image_name.replace('png', 'txt')

    with open(os.path.join(all_words_path, json_file_name), 'r') as f:
        all_words = json.load(f)

    with open(os.path.join(annotations_path, txt_file_name), 'r') as f:
        annotations = f.readlines()

    image_loc = os.path.join(images_path, image_name)

    # shutil.copy(image_loc, os.path.join(save_to,'images',image_name))

    image = Image.open(image_loc)
    w, h = image.size

    classes = [int(float(item.split()[0])) for item in annotations]
    annotations = [item.split()[1:] for item in annotations]
    
    annotations = [[float(item) for item in line ] for line in annotations]
    annotations = [denormalize(h, w, coord) for coord in annotations]

    all_words_tokenized = {}
    for word in all_words:
        # print(all_words[word]['text'])
        tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(all_words[word]['text']))
        all_words_tokenized[word] = {'tokens': tokens,
                                     'text' : all_words[word]['text'],
                                     'bbox' : all_words[word]['bbox']}

    # blocks_data = get_blocks(blocks, all_words_tokenized)
    
    

    #Split document as per the max token limit
    split_document =  list(chunks(all_words_tokenized, MAX_SEQ_LENGTH))  #[dict] * (ceil(len(all_words_tokenized) / MAX_SEQ_LENGTH))

    for doc_id, doc in enumerate(split_document):    
        # prepare and preprocess data for each split sub-document   
        out_json_obj = {}
        out_json_obj['blocks'] = {'first_token_idx_list': [], 'boxes': []}
        out_json_obj["words"] = []
        out_json_obj["parse"] = {"class": {}}
        for c in CLASSES:
            out_json_obj["parse"]["class"][c] = []
        out_json_obj["parse"]["relations"] = []

        num_tokens = 0

        ################################## Update tokens data ###################################
        for word in doc:
            word_text = doc[word]["text"]
            bb = doc[word]["bbox"]
            bb = [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]
            tokens = doc[word]['tokens']

            word_obj = {"text": word_text, "tokens": tokens, "boundingBox": bb}
            if len(word_text) != 0: # filter empty words
                out_json_obj["words"].append(word_obj)

            doc[word].update({'token_idx': num_tokens + 1}) 

            num_tokens += len(tokens)

            token_class = get_class(doc[word]['bbox'], annotations, classes)
            # print(doc[word]['text'], token_class)
            doc[word].update({'token_class': token_class})

            out_json_obj["parse"]["class"][token_class].append(doc[word]['tokens'])

        ############################################################################################
            
        ################################# Update blocks data #######################################
        blocks_data = get_blocks(blocks, doc, keep_blocks)

        for block in blocks_data:
            # ignore all empty blocks
            if blocks_data[block] != {} and len(blocks_data[block]['token_data']) > 0:
                # one more 'if' condition because somehow some values were getting appended more than once
                if not blocks_data[block]['token_data'][0]['token_idx'] in out_json_obj['blocks']['first_token_idx_list']:
                    out_json_obj['blocks']['first_token_idx_list'].append(blocks_data[block]['token_data'][0]['token_idx'])
                    out_json_obj['blocks']['boxes'].append(blocks_data[block]['block_data']['block_bbox'])
        ############################################################################################    
        
        ################################ Update linking data #######################################
        relation_data = get_links(doc, key_val_set)
        """
        Note: We could possibly break down KEY_VALUE_SET block from AWS into
        two blocks: KEY_BLOCK and VALUE_BLOCK. This is enable us to recreate 
        the block-wise linking used in FUNSD. 

        Also, linking is not necessary during inference. 
        """
        for item in relation_data:
            if 'key_token_idx' in item and 'value_token_idx' in item:
                relation_pair = [min(item['key_token_idx']), min(item['value_token_idx'])]
                out_json_obj["parse"]["relations"].append(relation_pair)
        ############################################################################################

        ################################## Update metadata #########################################
        out_json_obj["meta"] = {}

        save_image_name = image_name.replace('.png', f"_s_{doc_id}.png")
        save_json_name = save_image_name.replace('png', 'json')

        if image_name in train_set:
            out_json_obj["meta"]["image_path"] = os.path.join('Train', save_image_name)
            print(save_json_name, file=save_train_name)
            shutil.copy(os.path.join(images_path, image_name), 
                    os.path.join(train_save_path, save_image_name))

        elif image_name in test_set:
            out_json_obj["meta"]["image_path"] = os.path.join('Test', save_image_name)
            print(save_json_name, file=save_test_name)
            shutil.copy(os.path.join(images_path, image_name), 
                    os.path.join(test_save_path, save_image_name))
        out_json_obj["meta"]["imageSize"] = {"width": w, "height": h}
        out_json_obj["meta"]["voca"] = VOCA

        with open(os.path.join(ANNOTATION_SAVE_PATH, save_json_name), 'w') as f:
                json.dump(out_json_obj, f, indent=4)
        ############################################################################################

    
with open(
        os.path.join(OUTPUT_PATH, "class_names.txt"), "w", encoding="utf-8"
    ) as fp:
        fp.write("\n".join(CLASSES))

save_train_name.close()
save_test_name.close()


    

    