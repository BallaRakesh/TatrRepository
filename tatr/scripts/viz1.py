"""
Copyright (C) 2023 Microsoft Corporation

Assumes the data is in PASCAL VOC data format and the folder structure is:
[data_directory]/
- images/
- train/
- test/
- val/
"""

import argparse
import os
import json
from collections import defaultdict
import traceback
from align_boxes import align_coords,sort_coord
from tqdm import tqdm

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
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

color_map = defaultdict(lambda: ('magenta', 0, 1))
color_map.update({'table': ('brown', 0.1, 3), 'table row': ('blue', 0.04, 1),
                  'table column': ('red', 0.04, 1), 'table projected row header': ('cyan', 0.2, 3),
                  'table column header': ('magenta', 0.2, 3), 'table spanning cell': ('green', 0.6, 3)})

def plot_bbox(ax, bbox, color='magenta', linewidth=1, alpha=0):
    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                             edgecolor='none',facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                             edgecolor=color,facecolor='none',linestyle="--")
    ax.add_patch(rect) 
    
    
def read_json(filename):
    
    with open(filename, 'r') as f:
        tables = json.load(f)
        
    
    row_bbox = []
    col_bbox = []
    for table in tables:
        for item in table:
            if 'rows' in table:
                for row in table['rows']:
                    row_bbox.append(row['bbox'])
                    
            if 'columns' in table:
                for col in table['columns']:
                    col_bbox.append(col['bbox'])

    final_bboxes =  row_bbox + col_bbox 
    final_labels = ['table row']*len(row_bbox) + ['table column']*len(col_bbox)
    
    return final_bboxes, final_labels

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--images_dir',
                        help="Root directory for source data to process",
                        default='/home/lpt5355/table_extraction/sample_images',
                        required=False)
    
    parser.add_argument('--pascal_data_dir',
                        help="Root directory for source data to process",
                        default='/home/lpt5355/table_extraction/sample_labels',
                        required=False)
    parser.add_argument('--words_data_dir',
                         help="Root directory for source data to process")
    parser.add_argument('--split', default='',
                         help="Split to process")
    parser.add_argument('--output_dir',
                        help="Root directory for output data",
                        default='/home/lpt5355/table_extraction/sample1_viz',
                        required=False)
    return parser.parse_args()

def main():
    args = get_args()

    data_directory = args.pascal_data_dir
    images_directory = args.images_dir
    words_directory = args.words_data_dir
    split = args.split
    output_directory = args.output_dir
    #num_samples = args.num_samples

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    xml_filenames = [elem for elem in os.listdir(os.path.join(data_directory, split)) if elem.endswith(".xml")]
    
    if xml_filenames == []:
        xml_filenames = [item for item in os.listdir(os.path.join(data_directory, split)) if item.endswith('_structure.json')]

    for idx, filename in tqdm(enumerate(xml_filenames), 'Processing'):
        # if not num_samples is None and idx == num_samples:
        #     break
        # print(filename)
        
            xml_filepath = os.path.join(data_directory, split, filename)
            if xml_filenames[0].endswith('.xml'):
                img_name = filename.replace('xml', 'png')
            else:
                img_name = filename.replace('_structure.json', '.png')
            img_filepath = os.path.join(images_directory, img_name)
            
            if xml_filenames[0].endswith('.xml'):
                bboxes, labels = read_pascal_voc(xml_filepath)
            else:
                bboxes, labels = read_json(xml_filepath)
            img = Image.open(img_filepath).convert('RGBA')
            
            

            # TODO: Add option to include words
            #words_filepath = os.path.join(words_directory, filename.replace(".xml", "_words.json"))
            #try:
            #    with open(words_filepath, 'r') as json_file:
            #        words = json.load(json_file)
            #except:
            #    words = []
            
            ax = plt.gca()
            ax.imshow(img, interpolation="lanczos")
            plt.gcf().set_size_inches((24, 24))

            # tables = [bbox for bbox, label in zip(bboxes, labels) if label == 'table']
            # columns = [bbox for bbox, label in zip(bboxes, labels) if label == 'table column']
            # columns=sort_coord(columns)
            # columns=align_coords(columns,'column')
            
            
            
            rows = [bbox for bbox, label in zip(bboxes, labels) if label == 'table row']
            
            rows=sort_coord(rows)
            # rows=align_coords(rows,'row')
            column_headers = [bbox for bbox, label in zip(bboxes, labels) if label == 'table column header']
            projected_row_headers = [bbox for bbox, label in zip(bboxes, labels) if label == 'table projected row header']
            spanning_cells = [bbox for bbox, label in zip(bboxes, labels) if label == 'table spanning cell']

            
            # for column_num, bbox in enumerate(columns):
            #     if column_num % 2 == 0:
            #         linewidth = 2
            #         alpha = 0.25
            #         facecolor =(135/255, 0, 0)
            #         edgecolor = 'red'
            #         hatch = ''
            #     else:

            #         linewidth = 2
            #         alpha = 0.1
            #         facecolor = (1, 0,0)
            #         edgecolor = 'red'
            #         hatch = ''
            #     rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0, 
            #                              edgecolor=edgecolor, facecolor=facecolor, linestyle="-",
            #                              hatch=hatch, alpha=alpha)
            #     ax.add_patch(rect)
            #     rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
            #                                  edgecolor='red', facecolor='none', linestyle="-",
            #                                  alpha=0.8)
            #     ax.add_patch(rect)

            for row_num, bbox in enumerate(rows):
                
                if row_num % 3 == 0:
                    linewidth = 2
                    alpha = 0.6
                    edgecolor = 'blue'
                    facecolor = (7/255, 134/255, 24/255)
                    hatch = ''

                elif row_num%3==1:
                    linewidth = 2
                    alpha = 0.1
                    edgecolor = 'blue'
                    facecolor = (0, 1, 34/255)
                    hatch = ''

                else:
                    linewidth = 2
                    alpha = 0.1
                    facecolor = (143/255, 179/255, 0)
                    edgecolor = 'blue'
                    hatch = ''
                print(facecolor)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                         edgecolor=edgecolor, facecolor=facecolor, linestyle="-",
                                         hatch=hatch, alpha=0.1)
                ax.add_patch(rect)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                             edgecolor='blue', facecolor=facecolor, linestyle="-",
                                             alpha=0.1)
                ax.add_patch(rect)

            
            for bbox in column_headers:
                linewidth = 3
                alpha = 0.3
                facecolor = (0.3, 0.5, 0.5) #(0.5, 0.45, 0.25)
                edgecolor = (1, 0, 0.75) #(1, 0.9, 0.5)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                         edgecolor='none',facecolor=facecolor, alpha=alpha)
                ax.add_patch(rect)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0, 
                                         edgecolor=edgecolor,facecolor='none',linestyle="-", hatch='.')
                ax.add_patch(rect)

            for bbox in projected_row_headers:
                facecolor = (1, 0.9, 0.5) #(0, 0.75, 1) #(0, 0.4, 0.4)
                edgecolor = (1, 0.9, 0.5) #(0, 0.7, 0.95)
                alpha = 0.35
                linewidth = 3
                linestyle="--"
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                         edgecolor='none',facecolor=facecolor, alpha=alpha)
                ax.add_patch(rect)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1,
                                         edgecolor=edgecolor,facecolor='none',linestyle=linestyle)
                ax.add_patch(rect)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
                                         edgecolor=edgecolor,facecolor='none',linestyle=linestyle, hatch='\\\\')
                ax.add_patch(rect)

            for bbox in spanning_cells:
                color = (0.2, 0.5, 0.2) #(0, 0.4, 0.4)
                alpha = 0.4
                linewidth = 4
                linestyle="-"
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                         edgecolor='none',facecolor=color, alpha=alpha)
                ax.add_patch(rect)
                rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                         edgecolor=color,facecolor='none',linestyle=linestyle) # hatch='//'
                ax.add_patch(rect)

            # table_bbox = tables[0]
            # plt.xlim([table_bbox[0]-5, table_bbox[2]+5])
            # plt.ylim([table_bbox[3]+5, table_bbox[1]-5])
            plt.xticks([], [])
            plt.yticks([], [])

            legend_elements = [Patch(facecolor=(0,0,1), alpha=0.4,edgecolor='none',
                                     label='Row 1'),
                               Patch(facecolor=(0, 1,0), edgecolor='none',
                                     label='Row 2',alpha=0.3, hatch=''),
                               Patch(facecolor=(1, 127/255, 0), edgecolor='none',
                                     label='Row 3',alpha=0.4, hatch=''),
                               Patch(facecolor=(1, 0, 0),alpha=0.35, edgecolor='none',
                                     label='Column (odd)', hatch=''),
                               Patch(facecolor=(135/255, 0, 0), alpha=0.25,edgecolor='none',
                                     label='Column (even)'),
                            #    Patch(facecolor=(0.68, 0.8, 0.68), edgecolor=(0.2, 0.5, 0.2),
                            #          label='Spanning cell'),
                               Patch(facecolor=(0.3, 0.5, 0.5), edgecolor=(1, 0, 0.75),
                                     label='Column header', alpha=0.4,hatch='.'),
                            #    Patch(facecolor=(1, 0.965, 0.825), edgecolor=(1, 0.9, 0.5),
                            #          label='Projected row header', hatch='\\\\')]
            ]
            ax.legend(handles=legend_elements, bbox_to_anchor=(0, -0.02), loc='upper left', borderaxespad=0,
                         fontsize=16, ncol=4)  
            plt.gcf().set_size_inches(20, 20)
            plt.axis('off')
            if xml_filenames[0].endswith('.xml'):
                save_filepath = os.path.join(output_directory, filename.replace(".xml", "_ANNOTATIONS.jpg"))
            else:
                save_filepath = os.path.join(output_directory, filename.replace("_structure.json", "_ANNOTATIONS.jpg"))
            plt.savefig(save_filepath, bbox_inches='tight', dpi=150)
            #plt.show()
            plt.close()
        # except:
        #     traceback.print_exc()
        #     continue

if __name__ == "__main__":
    main()