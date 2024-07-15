from crop_tables import read_pascal_voc, generate_xml
import argparse
import os
from tqdm import tqdm
from PIL import Image
from fitz import Rect
import cv2

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--images_dir',
                        help="Path to the images folder")
    parser.add_argument('--ann_dir',
                        help="Path to the annotations folder")
    parser.add_argument('--out_dir',
                        help="Path to the where the cropped images must be saved to")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    images_dir = args.images_dir
    ann_dir = args.ann_dir
    out_dir = args.out_dir
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    images_list = images_dir

    xml_filenames = [elem for elem in os.listdir(ann_dir) if elem.endswith(".xml")]
    
    images_save_path = os.path.join(out_dir, 'images')
    os.makedirs(images_save_path, exist_ok=True)
    
    labels_save_path = os.path.join(out_dir, 'labels')
    os.makedirs(labels_save_path, exist_ok=True)

    locs_save_path = os.path.join(out_dir, 'logical_locs')
    os.makedirs(locs_save_path, exist_ok=True)

    for filename in tqdm(xml_filenames, 'Processing'):
        
        xml_filepath = os.path.join(ann_dir, filename)
        
        image_name = filename.replace('xml', 'png')
        # img_filepath = os.path.join(images_dir, image_name)
        
        if image_name in images_list:
            bboxes, labels = read_pascal_voc(xml_filepath)
            
            rows = [bbox for bbox, label in zip(bboxes, labels) if label == 'table row']
            cols = [bbox for bbox, label in zip(bboxes, labels) if label == 'table column']
            
            cells = []
            cell_locs = []
            for row_id, row in enumerate(rows):
                for col_id, col in enumerate (cols):
                    cell = Rect(row).intersect(Rect(col))
                    cells.append(list(cell))

                    row_start_id = row_id
                    row_end_id = row_id # need to add spanning info to update this
                    col_start_id = col_id
                    col_end_id = col_id # need to add spanning info to update this
                    cell_locs.append(row)

                    cell_locs.append([row_start_id, row_end_id, col_start_id, col_end_id])
            
            
            labels = ['cell'] * len(cells)          
            
            xml_obj = generate_xml(
                image_name,
                0,
                0,
                cells,
                labels
            )
            
            xml_obj.write(os.path.join(out_dir, filename))

            txtfilename = filename.replace('xml','txt')
            with open(os.path.join(locs_save_path, txtfilename), 'a+') as f:
                for item in cell_locs:
                    cell_loc_info = ','.join(item)
                    print(cell_loc_info, file=f)
            # img = Image.open(img_filepath)
            # img = cv2.imread(img_filepath)
            # for cell in cells:
            #     cell = [int(c) for c in cell]
            #     cv2.rectangle(img, cell, (0, 244, 0), 4)
            # cv2.imwrite('test.png', img)
            
            # exit()


if __name__ == '__main__':
    main()
