import os
import shutil
import argparse
from PIL import Image
from tqdm import tqdm
from gen_xml import generate_xml
from view_annotations import read_pascal_voc

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_data_dir',
                        help="Root directory for source data to process")
    parser.add_argument('--pascal_data_dir',
                        help="Root directory for source data to process")
    parser.add_argument('--output_dir',
                        help="Root directory for output data")
    parser.add_argument('--remove_classes_names', 
                        help= 'Name of all classes that needs to be removed, separated by commas')
    return parser.parse_args()

def main():
    args = get_args()
    
    imgs_dir = args.img_data_dir
    xml_dir = args.pascal_data_dir
    out_dir = args.output_dir
    remove_classes = args.remove_classes_names
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    save_images = os.path.join(out_dir, 'images')
    save_labels = os.path.join(out_dir, 'labels')
    os.makedirs(save_images, exist_ok=True)
    os.makedirs(save_labels, exist_ok=True)
    
    print(imgs_dir)
    print(xml_dir)
    print(out_dir)
    print(remove_classes)
    
    all_classes = ['column', 'row', 'column header', 'projected row header', 'spanning cell', 'trash']
    
    remove_classes = remove_classes.split(',')
    remove_classes = [item.strip() for item in remove_classes]
    
    for item in remove_classes:
        if item not in all_classes:
            raise ValueError(f"{all_classes} not present in classes list")
    
    xml_files = [elem for elem in os.listdir(xml_dir) if elem.endswith('.xml')]
    
    img_extensions = [item.split('.')[-1] for item in os.listdir(imgs_dir)]
    if len(set(img_extensions)) != 1:
        raise TypeError("Multiple image extensions detected in image set")
    
    img_extension = img_extensions[0]
        
    for xml_file in tqdm(xml_files, 'Preparing'):
        filename = xml_file
        imgname = filename.replace('xml',img_extension)
        
        bboxes, labels = read_pascal_voc(os.path.join(xml_dir, xml_file))
        
        required_bboxes = []
        required_labels = []
        for cls in all_classes:
            if cls not in remove_classes:
                if 'table' not in cls and cls != 'projected row header' and cls != 'trash':
                    cls = f"table {cls}"
                cls_bboxes = [bbox for bbox, label in zip(bboxes, labels) if label == cls]
                cls_labels = [cls] * len(cls_bboxes)
                required_bboxes.extend(cls_bboxes)
                required_labels.extend(cls_labels)
                
                
        img = Image.open(os.path.join(imgs_dir, imgname))
        table_w, table_h = img.size
        
        xml_obj = generate_xml(
            imgname=imgname,
            table_h=table_h,
            table_w=table_w,
            bboxes=required_bboxes, 
            labels=required_labels,
        )
        
        xml_obj.write(os.path.join(save_labels, filename))
        shutil.copy(
            os.path.join(imgs_dir, imgname),
            os.path.join(save_images, imgname)
        )
        
        # img.save(os.path.join(images_save_path, image_name))
        
        
if __name__ == '__main__':
    main()