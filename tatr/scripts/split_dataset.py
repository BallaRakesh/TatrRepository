import os
import shutil
import argparse
from PIL import Image
from tqdm import tqdm
from gen_xml import generate_xml
from view_annotations import read_pascal_voc



def clean_file_list(file_list):
    file_list = [item.replace('\n','') for item in file_list]
    # file_list = [item.replace(ext, '') 
    for i in range(len(file_list)): 
        if 'png' in file_list[i]:
            ext='.png' 
        else :
            ext='.xml'
            
        file_list[i] = file_list[i].replace(ext,'').strip()
        
    return file_list


def store_data(
            all_img_path: str,
            all_label_path: str,
            save_img_path: str,
            save_label_path: str,
            save_items: list
        ):
    
    filetxt_path = '/'.join(save_img_path.split('/')[:-1])
    filetxt_name = f"{save_img_path.split('/')[-2]}_filelist.txt"
        
    for item in save_items:
        
        with open(os.path.join(filetxt_path, filetxt_name), 'a+') as f:
            if f"{item}.png" in os.listdir(all_img_path) and f"{item}.xml" in  os.listdir(all_label_path):
                
                shutil.copy(
                    os.path.join(all_img_path, f"{item}.png"),
                    os.path.join(save_img_path, f"{item}.png")
                )
                
                shutil.copy(
                    os.path.join(all_label_path, f"{item}.xml"),
                    os.path.join(save_label_path, f"{item}.xml")
                )
                
                
                
                print(f"{item}.xml", file=f)
    

def generate_train_val_split(train_txt_path, val_split_ratio=0.2):
    with open(train_txt_path, 'r') as f:
        all_train_files = f.readlines()
    
    
    all_train_files = clean_file_list(all_train_files)
    
    
    train_max_idx = int(len(all_train_files) * (1 - val_split_ratio))
    
    train_files = all_train_files[:train_max_idx]
    val_files = all_train_files[train_max_idx:]
    
    return train_files, val_files

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root_dir',
                        help="Root directory for source data to process")
    return parser.parse_args()

def main():
    args = get_args()
    
    root_dir = args.root_dir
    
    all_images_dir = os.path.join(root_dir, 'images')
    all_labels_dir = os.path.join(root_dir, 'labels')
    
    train_dir = os.path.join(root_dir, 'train')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir, exist_ok=True)
    
    val_dir = os.path.join(root_dir, 'val')
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir, exist_ok=True)
    
    test_dir = os.path.join(root_dir, 'test')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    
    train_images_dir = os.path.join(train_dir, 'images')
    os.makedirs(train_images_dir, exist_ok=True)
    
    train_labels_dir = os.path.join(train_dir, 'labels')
    os.makedirs(train_labels_dir, exist_ok=True)
    
    val_images_dir = os.path.join(val_dir, 'images')
    os.makedirs(val_images_dir, exist_ok=True)
    
    val_labels_dir = os.path.join(val_dir, 'labels')
    os.makedirs(val_labels_dir, exist_ok=True)
    
    test_images_dir = os.path.join(test_dir, 'images')
    os.makedirs(test_images_dir, exist_ok=True)
    
    test_labels_dir = os.path.join(test_dir, 'labels')
    os.makedirs(test_labels_dir, exist_ok=True)
    
    
    if 'val.txt' not in os.listdir(root_dir):
        train_list, val_list = generate_train_val_split(os.path.join(root_dir, 'train.txt'))
        with open(os.path.join(root_dir, 'val.txt'), 'w') as f:
            for item in val_list:
                print(f"{item}.xml", file=f)
        os.remove(os.path.join(root_dir, 'train.txt'))
        with open(os.path.join(root_dir, 'train.txt'), 'w') as f:
            for item in train_list:
                print(f"{item}.xml", file=f)
    else:
        with open(os.path.join(root_dir, 'val.txt'), 'r') as f:
            val_list = f.readlines()
            
        val_list = clean_file_list(val_list)
    
    with open(os.path.join(root_dir, 'train.txt'), 'r') as f:
            train_list = f.readlines()
            
    train_list = clean_file_list(train_list)
    
    with open(os.path.join(root_dir, 'test.txt'), 'r') as f:
            test_list = f.readlines()
            
    test_list = clean_file_list(test_list)
    
    
    store_data(all_images_dir, all_labels_dir, train_images_dir, train_labels_dir, train_list)
    store_data(all_images_dir, all_labels_dir, test_images_dir, test_labels_dir, test_list)
    store_data(all_images_dir, all_labels_dir, val_images_dir, val_labels_dir, val_list)
    
    
    print(f"Images, labels in train set: {len(os.listdir(train_images_dir))}, {len(os.listdir(train_labels_dir))}")
    print(f"Images, labels in val set: {len(os.listdir(val_images_dir))}, {len(os.listdir(val_labels_dir))}")
    print(f"Images, labels in test set: {len(os.listdir(test_images_dir))}, {len(os.listdir(test_labels_dir))}")
    
if __name__ == '__main__':
    main()