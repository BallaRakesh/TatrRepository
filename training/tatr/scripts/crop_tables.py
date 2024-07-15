import os 
import argparse
from PIL import Image, ImageDraw
from tqdm import tqdm
import xml.etree.ElementTree as ET

def read_pascal_voc(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    labels = []

    for object_ in root.iter('object'):

        ymin, xmin, ymax, xmax = None, None, None, None
        
        label = object_.find("name").text

        for box in object_.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)

        bbox = [xmin, ymin, xmax, ymax] # PASCAL VOC
        
        bboxes.append(bbox)
        labels.append(label)

    return bboxes, labels

def update_bboxes(table_w, table_h, pad_val, bboxes, labels):
    x_diff, y_diff = [bboxes[i] for i in range(len(bboxes)) if labels[i] == 'table'][0][:2]
    
    x_diff -= pad_val
    y_diff -= pad_val
    
    for i in range(len(bboxes)):
        # print(bboxes[i], '\t', labels[i])
        bboxes[i] = [float(bboxes[i][0] - x_diff), float(bboxes[i][1] - y_diff), 
                     float(bboxes[i][2] - x_diff), float(bboxes[i][3] - y_diff)]
        # print(bboxes[i], '\t', labels[i],  end='\n\n')
    return bboxes

def indent(elem, level=0):
    i = "\n" + level*"  "
    j = "\n" + (level-1)*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem        


def generate_xml(xmlname: str, 
                 table_w: int, 
                 table_h: float, 
                 bboxes: list, 
                 labels: list):
    root = ET.Element('annotations')
    
    tree = ET.ElementTree(root)
    
    # folder = ET.SubElement(root)
    filename = ET.SubElement(root, 'filename')
    filename.text = xmlname
    
    source = ET.SubElement(root, 'source')
    db = ET.SubElement(source, 'database')
    db.text = 'table_extraction'
    annotation = ET.SubElement(source, 'annotation')
    annotation.text = 'Unknown'
    img = ET.SubElement(source, 'image')
    img.text = 'Unknown'
    
    img_size = ET.SubElement(root, 'size')
    img_width = ET.SubElement(img_size, 'width')
    img_width.text = str(table_w)
    img_height = ET.SubElement(img_size, 'height')
    img_height.text = str(table_h)
    img_depth = ET.SubElement(img_size, 'depth')
    
    segmented = ET.SubElement(source, 'segmented')
    segmented.text = '0'
        
    for i in range(len(bboxes)):
        object = ET.SubElement(root, 'object')
        
        obj_name = ET.SubElement(object, 'name')
        obj_name.text = labels[i]
        obj_truncated = ET.SubElement(object, 'truncated')
        obj_truncated.text = '0' 
        obj_occluded = ET.SubElement(object, 'occluded')
        obj_occluded.text = '0'
        obj_difficult = ET.SubElement(object, 'difficult')
        obj_difficult.text = '0'
        
        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(bboxes[i][0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(bboxes[i][1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(bboxes[i][2])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(bboxes[i][3])
    
    indent(root)
    return tree

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--images_dir',
                        help="Path to the images folder")
    parser.add_argument('--ann_dir',
                        help="Path to the annotations folder")
    parser.add_argument('--pad_val', default=0, 
                        help="Pad image by how many pixels")
    parser.add_argument('--out_dir',
                        help="Path to the where the cropped images must be saved to")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    images_dir = args.images_dir
    ann_dir = args.ann_dir
    out_dir = args.out_dir
    pad_val = float(args.pad_val)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    xml_filenames = [elem for elem in os.listdir(ann_dir) if elem.endswith(".xml")]
    
    images_save_path = os.path.join(out_dir, 'images')
    os.makedirs(images_save_path, exist_ok=True)
    
    labels_save_path = os.path.join(out_dir, 'labels')
    os.makedirs(labels_save_path, exist_ok=True)

    for filename in tqdm(xml_filenames, 'Processing'):
        
        xml_filepath = os.path.join(ann_dir, filename)
        
        image_name = filename.replace('xml', 'png')
        img_filepath = os.path.join(images_dir, image_name)
        
        bboxes, labels = read_pascal_voc(xml_filepath)
        img = Image.open(img_filepath)
        tables = [bbox for bbox, label in zip(bboxes, labels) if label == 'table']
        
        
        if tables != []:
            tables = tables[0]
            
            tables = [tables[0]-pad_val, tables[1]-pad_val, tables[2]+pad_val, tables[3]+pad_val]
            
            cropped_table = img.crop(tables)
            # img1 = ImageDraw.Draw(cropped_table)
            new_w, new_h = cropped_table.size

            bboxes = update_bboxes(new_w, new_h, pad_val, bboxes, labels)
            
            # for bbox in bboxes:
            #     img1.rectangle(bbox, outline ="red")
            # cropped_table.save(os.path.join('viz', image_name))
            # # exit()
            
            xml_object = generate_xml(filename, new_w, new_h, bboxes, labels)
            xml_object.write(os.path.join(labels_save_path, filename))
            
            cropped_table.save(os.path.join(images_save_path, image_name))
        
if __name__ == '__main__':
    main()