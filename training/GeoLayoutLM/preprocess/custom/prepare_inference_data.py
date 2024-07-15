import os 
import json
from google.cloud import vision
from base64 import b64encode
import argparse
import shutil
from PIL import Image
from transformers import BertTokenizer
import numpy as np
from math import dist
import cv2
import statistics as st

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--images_dir',
                        help="Directory where the images to process are")
    parser.add_argument('--ocr_dir',
                        help="Directory where the OCR content are")
    parser.add_argument('--dump_dir',
                        help="Path to where the generated data must be dumped to")
    
    return parser.parse_args()

# OCR Vision function
def get_ocr_vision_api(jsonname, file):
    image = file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "digital-hall-399509-cae6bb37c802.json"
    ctxt = b64encode(image.read()).decode()
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=ctxt)

    response = client.text_detection(image=image)

    # for res in response.text_annotations:
    # 	print(res.confidence)

    word_coordinates = []
    for i, text in enumerate(response.text_annotations):
        if i != 0:
            vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
            x1 = min([v.x for v in text.bounding_poly.vertices])
            x2 = max([v.x for v in text.bounding_poly.vertices])
            y1 = min([v.y for v in text.bounding_poly.vertices])
            y2 = max([v.y for v in text.bounding_poly.vertices])
            if x2 - x1 == 0:
                x2 += 1
            if y2 - y1 == 0:
                y2 += 1
            """"left": x1,
                "top": y1,
                "width": x2 - x1,
                "height": y2 - y1,"""
            word_coordinates.append({
                "word": text.description,
                "left": x1,
                "top": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })
        else:
            all_text = text.description
    save_coords = {}
    for i in range(len(word_coordinates)):
        save_coords[i] = {
            "text": word_coordinates[i]['word'],
            "score": 0.0,
            "left": word_coordinates[i]['left'],
            "top": word_coordinates[i]['top'],
            "width": word_coordinates[i]['width'],
            "height": word_coordinates[i]['height'],
            "x1": word_coordinates[i]['x1'],
            "y1": word_coordinates[i]['y1'],
            "x2": word_coordinates[i]['x2'],
            "y2": word_coordinates[i]['y2'],
            "bbox": [word_coordinates[i]['x1'], word_coordinates[i]['y1'], 
                    word_coordinates[i]['x2'], word_coordinates[i]['y2']]
        }
    with open(f"/New_Volume/number_theory/GeoLayoutLM/dataset/test_ocr/{jsonname}", 'w') as f:
        json.dump(save_coords, f, indent=4)
    # im_name = list(file.split('/'))[-1]
    return word_coordinates


def read_ocr( file):
    # json_path = '/New_Volume/number_theory/dataset/ocr_pad'
    # json_name = file.replace('png','json')
    print("called Image OCR...")
    word_coordinates = []

    ocr_path = os.path.join(file)
    # print(ocr_path)
    with open(ocr_path, 'r') as f:
        ocr_data = json.load(f)
    for data in ocr_data:
        word_coordinates.append({
            'word': ocr_data[data]['text'],
            'x1': ocr_data[data]['x1'],
            'y1': ocr_data[data]['y1'],
            'x2': ocr_data[data]['x2'],
            'y2': ocr_data[data]['y2']
        })

    return word_coordinates

def sort_coords(coords, axis=None):
    if axis == 1:
        points = [[item['bbox'][1], item['bbox'][3]] for item in coords]
    elif axis == 0:
        points = [[item['bbox'][0], item['bbox'][2]] for item in coords]
    else:
        points = [item['bbox'][:2] for item in coords]
    points_dist = []
    for coord in points:
        points_dist.append(round(dist((0, 0), coord), 2))
    sorted_order = np.argsort(points_dist)
    sorted_coords = [coords[i] for i in sorted_order]
    return sorted_coords

def get_lines(font_size, coords):
    line_num = 0
    lines = {}
    for i in range(len(coords)-1):
        word_dist = coords[i+1]['bbox'][1] - coords[i]['bbox'][1]
        if word_dist <= font_size:
            if not line_num in lines:
                lines.update({line_num: [coords[i]]})
            else:
                lines[line_num].append(coords[i])
        else:
            if not line_num in lines:
                lines.update({line_num: [coords[i]]})
            else:
                lines[line_num].append(coords[i])
            line_num += 1
            # lines.update({line_num: [coords[i+1]]})
    lines[len(lines)-1].append(coords[i+1])
    for item in lines:
        x_min = min([x['bbox'][0] for x in lines[item]])
        x_max = max([x['bbox'][2] for x in lines[item]])
        
        y_min = min([y['bbox'][1] for y in lines[item]])
        y_max = max([y['bbox'][3] for y in lines[item]])
        lines[item] = {
            'block_bbox': [x_min, y_min, x_max, y_max],
            'words': lines[item]
        }
        
    return lines

def get_line_num(lines, start_num):
    for line in range(start_num, -1, -1):
        if lines[line] != None:
            return line

def get_seperators(lines, min_word_threshold):
    for line in lines:
        if abs(lines[line]['block_bbox'][2]-lines[line]['block_bbox'][0]) <= min_word_threshold:
            if line == 0:
                continue
            merge_with_line = get_line_num(lines, line-1)  
            print(line,'merge with', merge_with_line)      
            lines[merge_with_line]['words'].append(lines[line]['words'])
            lines[line] = None
            
    blocks = {}
    for line in lines:
        if lines[line] != None:
            blocks[line] = lines[line]
    return blocks
            
def get_block_box(words):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for word in words:
        # print(word)
        xmin.append(word['x1'])
        ymin.append(word['y1'])
        xmax.append(word['x2'])
        ymax.append(word['y2'])
    return [min(xmin), min(ymin), max(xmax), max(ymax)]

def get_label(annotation_dir):
    """
    Fetches the labels details from the annotation for evaluation
    or training data preparation

    Args:
        annotation_dir (str): path to the pascal voc table annotation director
    """
    

def get_blocks(imgname, coords):
    """
    Each row (words with same y-min) will become a block
    """
    for coord in coords:
        coord.update({'bbox': [coord['x1'], coord['y1'], coord['x2'], coord['y2']]})
    # coords = [[item['x1'], item['y1'], item['x2'], item['y2']] for item in coords] 
    coords = sort_coords(coords, axis=1)
    
    font_size = round(np.mean([coord['y2']-coord['y1'] for coord in coords ]))
    print(f"Font size: {font_size}")
    
    lines = get_lines(font_size, coords)
    
    line_width = [lines[line]['block_bbox'][2]-lines[line]['block_bbox'][0] for line in lines ]
    # line_width = sorted(line_width, reverse=True)
    # min_width_threshold = line_width[0] * 0.9 #int(st.mode(line_width) * 0.9)
    # print(line_width, min_width_threshold)
    # blocks = get_seperators(lines, min_width_threshold)
    # return blocks
    # exit()
    return lines
    
    lines_with_max_widths = [1 if width > max(line_width)*0.85 else 0 for width in line_width ]
    lines_with_max_widths[0] = 1
    print(f"Lines with max widths {lines_with_max_widths}")
    
    numbers_in_line = [0 for i in range(len(lines))]
    for line in lines:
        for word in lines[line]['words']:
            try:
                num = float(word['word'])
                numbers_in_line[line] += 1
            except:
                continue
    print(f"Numbers in line: {numbers_in_line}")     
    line_distances = []
    for ln in lines:
        if not ln == len(lines)-1:
            line_distances.append(lines[ln+1]['block_bbox'][1] - lines[ln]['block_bbox'][3])
    line_distances.append(0)
    if not line_distances == []:
        # print(f"Line distances: {line_distances}")
        
        # if st.mode(line_distances) < font_size:
        #     print(f"Line distances: {line_distances}")
        
        merged_lines = {}
        n_blocks = 0
        for i in range(len(lines_with_max_widths)):
            if lines_with_max_widths[i] == 1 :
                merged_lines.update({n_blocks: lines[i]})
                n_blocks += 1
            elif line_distances[i] > font_size:# and lines_with_max_widths == 0 :
                merged_lines.update({n_blocks: lines[i+1]})
                n_blocks += 1
            # elif numbers_in_line[i] > 2:
            #     merged_lines.update({n_blocks: lines[i]})
            #     n_blocks += 1
        # for ln in merged_lines:
        #     print(ln, merged_lines[ln])
            
        # merged_lines = {}
        # n_blocks = 0
        # for ln in range(len(line_distances)):
        #     # print(f"Line width: {line_width[ln], 0.98*line_width[ln+1]}")
        #     if line_distances[ln] > font_size or lines_with_max_widths[ln] == 1:
        #         n_blocks+=1
        #         merged_lines.update({n_blocks: lines[ln]})
                
        #     else:
        #         if n_blocks in merged_lines:
        #             merged_lines[n_blocks]['words'].extend(lines[ln]['words'])
        #             new_block_bbox = get_block_box(merged_lines[n_blocks]['words'])
        #             merged_lines[n_blocks]['block_bbox'] = new_block_bbox
        # ln += 1            
        # if line_distances[ln-1] > font_size or lines_with_max_widths[ln] == 1:
        #         n_blocks+=1
        #         merged_lines.update({n_blocks: lines[ln]})
                
        # else:
        #     if n_blocks in merged_lines:
        #         merged_lines[n_blocks]['words'].extend(lines[ln]['words'])
        #         new_block_bbox = get_block_box(merged_lines[n_blocks]['words'])
        #         merged_lines[n_blocks]['block_bbox'] = new_block_bbox
               
        # ln += 1
        # if line_distances[ln-1] > font_size or lines_with_max_widths[ln-1] == 1:
        #         if ln in merged_lines:
        #             merged_lines[ln].append(lines[ln]['words'])
        #             merged_lines[ln]['block_bbox'] = [merged_lines[ln]['block_bbox'][0], merged_lines[ln]['block_bbox'][1],
        #                                               lines[ln]['block_bbox'][2], lines[ln]['block_bbox'][3]]
        #         else:
        #             merged_lines.update({ln: lines[ln]})
        #             n_blocks+=1
        # print(lines[ln+1])
        # print(f"Number of blocks: {n_blocks}")
        # for blk in range(len(lines)-1):
        #     for ln in range(1, len(lines)):
        #         if lines[ln] != None and lines[blk] != None:
        #             if lines[blk]['block_bbox'][3] - lines[ln]['block_bbox'][1] < font_size:
        #                 if not blk in merged_lines:
        #                     merged_lines.update({blk: lines[blk]})
        #                 merged_lines[blk]['words'].append(lines[ln]['words'])
        #                 # merged_lines[blk]['block_bbox'][-2:] = lines[ln]['block_bbox'][-2:]
        #                 lines[ln] = None
        
        # for ln in range(len(lines)-1):
        #     if not lines[ln] == None and (lines[ln]['block_bbox'][3] - lines[ln+1]['block_bbox'][1]) < font_size:
                
        #         merge_line_num = get_line_num(lines, ln)
        #         print(ln, merge_line_num)
                
        #         lines[merge_line_num]['words'].append(lines[ln+1]['words'])
        #         lines[merge_line_num]['block_bbox'][-2:] = lines[ln+1]['block_bbox'][-2:]
        #         lines[ln+1]['words'] = None
        # blocks = {}
        # for ln in lines:
        #     if not lines[ln] == None:
        #         print(f"See: {lines[ln]}")
        #         blocks.update({ln: lines[ln]})
        # for ln in range(len(lines)-1):
        #     print(lines[ln]['block_bbox'], lines[ln+1]['block_bbox'])
        # # exit()
        return merged_lines
    # # data_max_padding = 1
    # # # if len(line_distances) > 1:
    # # #     header_data_padding = line_distances[0]
    # # #     data_max_padding = max(line_distances[1:])
    # # # else:
    # # #     header_data_padding = 0
    # # #     data_max_padding = 0
    # # block_num = 0
    # # blocks = {}
    # # dump_ld = open('line_distances.txt','a+') 
    # # print(f"{imgname} ::::::::: {line_distances}", file=dump_ld)
    # # dump_ld.close()
    # # sorted_line_distances = sorted(line_distances, reverse=True)
    # # print(sorted_line_distances)
    # # if len(sorted_line_distances) > 2:
    # #     pos = 2
    # # else:
    # #     pos = 0
    # # for i in range(len(line_distances)):
    # #     if line_distances[i] < sorted_line_distances[pos]:
    # #         if block_num in blocks:
    # #             blocks[block_num].append(lines[i])
    # #             blocks[block_num].append(lines[i+1])
    # #         else:
    # #             blocks[block_num] = [lines[i], lines[i+1]]
    # #     else:
    # #         if block_num in blocks:
    # #             blocks[block_num].append(lines[i])
    # #         else:
    # #             blocks[block_num] = [lines[i]]
    # #         block_num += 1
    # #         blocks[block_num] = [lines[i+1]]
    
    # # # for i in range(len(line_distances)):
        
    # # #     if line_distances[i] > 5 : #or abs(header_data_padding -line_distances[i]) <= 2:
    # # #         if block_num in blocks:
    # # #             blocks[block_num].append([lines[i]])
    # # #         else:
    # # #             blocks[block_num] = [lines[i]]
    # # #         block_num+=1
    # # #         blocks[block_num] = [lines[i+1]] 
    # # #     else:
    # # #         # if block_num in blocks:
    # # #         #     blocks[block_num].append(lines[i])
    # # #         # else:
    # # #         #     
    # # #         blocks[block_num].append([lines[i]])
    # # #         blocks[block_num].append([lines[i+1]])
            
    # # for block in blocks:
    # #     if len(blocks[block])>1:
    # #         xmin, ymin, xmax, ymax = [], [], [], []
    # #         for pts in blocks[block]:
    # #             xmin.append(pts[0])
    # #             ymin.append(pts[1])
    # #             xmax.append(pts[2])
    # #             ymax.append(pts[3])
    # #         blocks[block] = [min(xmin), min(ymin), max(xmax), max(ymax)]
    # #     else:
    # #         blocks[block] = blocks[block][0]
            
    # # return blocks
                

def main():
    args = get_args()

    images_path = args.images_dir
    ocr_path = args.ocr_dir
    dump_path = args.dump_dir
    
    # if os.path.exists(dump_path):
    #     shutil.rmtree(dump_path)
    
    # if os.path.exists(os.path.join(dump_path, 'preprocessed_files_val.txt')):
    #     shutil.rmtree(os.path.join(dump_path, 'preprocessed_files_val.txt'))
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
        # os.makedirs(os.path.join(dump_path, 'training_data'))
        # os.makedirs(os.path.join(dump_path, 'training_data', 'images'))
        # os.makedirs(os.path.join(dump_path, 'training_data', 'annotations'))
        os.makedirs(os.path.join(dump_path, 'testing_data'))
        os.makedirs(os.path.join(dump_path, 'testing_data', 'images'))
        os.makedirs(os.path.join(dump_path, 'testing_data', 'annotations'))
    
    VOCA = "bert-base-uncased"
    
    tokenizer = BertTokenizer.from_pretrained(VOCA, do_lower_case=True)
    
    images_list = os.listdir(images_path)
    ocr_list = os.listdir(ocr_path)
    CLASSES = ['data_cell', 'header_cell', 'trash', 'O']
    
    for imgname in images_list:
        print(imgname)
        jsonname = imgname.replace('png','json')
        if ocr_path != None and jsonname in ocr_list :
            coords = read_ocr(os.path.join(ocr_path, jsonname))
        else:
            img = open(os.path.join(images_path, imgname), 'rb')
            coords = get_ocr_vision_api(jsonname, img)
            img.close()
        
        
        blocks = get_blocks(imgname, coords)
        
        
        if blocks != None:
            img = cv2.imread(os.path.join(images_path, imgname))
            for item in blocks:
                coords = blocks[item]['block_bbox']
                cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 244), 1)
            cv2.imwrite(os.path.join('block_viz', imgname), img)
            
        # # prepare_data_struct = {
        # #     "meta": {
        # #     "image_path": str,
        # #     "imageSize": {
        # #         "width": int,
        # #         "height": int
        # #         },
        # #         "voca": str
        # #     },
        # #     "blocks": {
        # #         "first_token_idx_list": [],
        # #         "boxes": [],
        # #     },
        # #     "words": [],
        # #     'parse': {
        # #         'class': {},
        # #         'relations': []
        # #     }
        # # }
        
        # # for c in CLASSES:
        # #     prepare_data_struct["parse"]['class'].update({c: []})
        
        # # image_file = os.path.join(images_path, imgname)
        # # image_w, image_h = Image.open(image_file).size
        
        # # prepare_data_struct['meta']['image_path'] = image_file
        # # prepare_data_struct['meta']['imageSize']['width'] = image_w
        # # prepare_data_struct['meta']['imageSize']['height'] = image_h
        # # prepare_data_struct['meta']['voca'] = "bert-base-uncased"

        # # num_tokens = 0
        
        
        # # for block in blocks:
        # #     real_word_idx = 0
        # #     prepare_data_struct['blocks']['boxes'].append(blocks[block]['block_bbox'])
        # #     for coord in blocks[block]['words']:
        # #         word_text = coord["word"]
        # #         bb = [coord["x1"], coord["y1"], coord["x2"], coord["y2"]]
        # #         bb = [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]
        # #         tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_text))
            
        # #         word_obj = {"text": word_text, "tokens": tokens, "boundingBox": bb}
        # #         prepare_data_struct['words'].append(word_obj)
                    
        # #         if real_word_idx == 0:
        # #             prepare_data_struct['blocks']['first_token_idx_list'].append(num_tokens+1)
        # #         num_tokens+=len(tokens)
        # #         real_word_idx += 1
            
                
        
        # # # for item in coords:
        # # #     word_text = item["word"]
        # # #     bb = [item["x1"], item["y1"], item["x2"], item["y2"]]
        # # #     bb = [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]
        # # #     tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_text))
        
        # # #     word_obj = {"text": word_text, "tokens": tokens, "boundingBox": bb}
        # # #     prepare_data_struct['words'].append(word_obj)
        
                
        # # #     prepare_data_struct['blocks']['first_token_idx_list'].append(num_tokens+1)
        # # #     num_tokens+=len(tokens)
        # # #     prepare_data_struct['blocks']['boxes'].append([item["x1"], item["y1"], item["x2"], item["y2"]])
            
        # # shutil.copy(image_file,
        # #             os.path.join(dump_path, 'testing_data', 'images', imgname))
        
        # # with open(os.path.join(dump_path, 'testing_data', 'annotations', jsonname), 'w') as f:
        # #     json.dump(prepare_data_struct, f, indent=4)
        
        # # with open(os.path.join(dump_path, 'preprocessed_files_val.txt'), 'a+') as f:
        # #     print(os.path.join('testing_data', 'annotations', jsonname), file=f)
            
if __name__ == '__main__':
    main()