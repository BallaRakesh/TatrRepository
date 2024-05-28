# import cv2
import os 
import xml.etree.ElementTree as ET
import numpy as np
# import shutil
from write_gt import GroundTruthXML
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "-xml",
    "--xml_dir",
    help="Path to XML Files",
    # required=True,
    default='../final_dataset_26_sep_23/train/labels',
)

parser.add_argument(
    "-o",
    "--output_path",
    help="path to the output directory",
    # required=True,
    default='../final_dataset_26_sep_23/train/rc_labels'
)
configs = parser.parse_args()

xml_dir = configs.xml_dir
output_dir= configs.output_path

os.makedirs(output_dir,exist_ok=True)

files_with_multiple_tables =[]
# xml_files =['IM-000000010733583-AP1.xml', 'IM-000000009702051-AP1.xml', 'IM-000000010652244-AP1.xml', 'IM-000000010640299-AP1.xml', 'IM-000000010912719-AP1.xml', 'IM-000000011101915-AP1.xml', 'IM-000000010534569-AP1.xml', 'IM-000000010394960-AP9.xml', 'IM-000000010667231-AP1.xml', 'IM-000000011188935-AP1.xml', 'IM-000000010652236-AP1.xml', 'IM-000000010662988-AP1.xml']
# print("+++++++",len(xml_files))
xml_files=os.listdir(xml_dir)
for xml_file in xml_files:
    print('CONVERTING : ',xml_file)
    tree = ET.parse(os.path.join(xml_dir,xml_file))
    root=tree.getroot()
    fileName=root.find('filename').text
    size = root.find('size')
    width=size.find('width').text
    height=size.find('height').text
    xml_out=GroundTruthXML(filename=fileName,Width=width,Height=height)

    rows_list=defaultdict(list)
    columns_list=defaultdict(list)
    spanningCell_list=defaultdict(list)
    tables= root.findall("./object/[name='table']")
    # img_path = '/home/ntlpt-52/work/IDP/Table_Extraction/New_Annotated_Data/Split_Model_Reannotated/images/'+xml_file.replace('xml','png')
    # cv2.imshow('img',cv2.imread(img_path))
    # cv2.waitKey(0)

    if len(tables)>1:
        files_with_multiple_tables.append(xml_file)
        continue
    for table in tables:
        name = table.find('name').text
        bounding_box = table.find('bndbox')
        xmin=bounding_box.find('xmin').text
        ymin=bounding_box.find('ymin').text
        xmax=bounding_box.find('xmax').text
        ymax=bounding_box.find('ymax').text
        xml_out.add_table(xmin_val=xmin,ymin_val=ymin,xmax_val=xmax,ymax_val=ymax)
        table={'xmin_val':xmin,'ymin_val':ymin,'xmax_val':xmax,'ymax_val':ymax}
        # print('TABLE :',table)
        
    for child in root:
        if child.tag=='object':           
            name = child.find('name').text

            bounding_box = child.find('bndbox')
            xmin=bounding_box.find('xmin').text
            ymin=bounding_box.find('ymin').text
            xmax=bounding_box.find('xmax').text
            ymax=bounding_box.find('ymax').text
               
            if name =='table column':
                columns_list['xmin'].append(int(eval(xmin)))
                columns_list['xmax'].append(int(eval(xmax)))
                columns_list['xmin_text'].append(xmin)
                columns_list['xmax_text'].append(xmax)
                # print('table_column : xmin - ',xmin,' ymin - ',ymin,' xmax - ',xmax,' ymax- ',ymax)
                # if eval(table['xmax_val'])-eval(xmax)<5 or eval(table['xmax_val'])<eval(xmax):
                    # continue
                # print("adding column")
                xml_out.add_column(xmin_val=xmax,ymin_val=ymin,xmax_val=xmax,ymax_val=ymax)
            elif name =='table row':
                rows_list['ymin'].append(int(eval(ymin)))
                rows_list['ymax'].append(int(eval(ymax)))
                rows_list['ymin_text'].append(ymin)
                rows_list['ymax_text'].append(ymax)
                # print('table_row : xmin - ',xmin,' ymin - ',ymin,' xmax - ',xmax,' ymax- ',ymax)
                if eval(table['ymax_val'])-eval(ymax)<5 or eval(table['ymax_val'])<eval(ymax):
                    continue
                # print('Adding row')
                xml_out.add_row(xmin_val=xmin,ymin_val=ymax,xmax_val=xmax,ymax_val=ymax)
            # elif name =='table spanning cell':
            #     spanningCell_list['xmin_ymin'].append((int(eval(xmin)),int(eval(ymin))))
            #     spanningCell_list['xmax_ymax'].append((int(eval(xmax) ),int(eval(ymax))))
            #     spanningCell_list['text'].append({'xmax_val':xmax,'ymax_val':ymax})
 
    dont_care = np.full((len(rows_list['ymin']),len(columns_list['xmin'])),"False")
    # print('rows: ',rows_list)
    for i in range(0,len(rows_list['ymin'])):
        start_row=i
        ymin_value = rows_list['ymin'][i]
        for j in range(0,len(columns_list['xmin'])):
            ymax_value=rows_list['ymax'][i]
            xmin_value=columns_list['xmin'][j]
            xmax_value=columns_list['xmax'][j]
            xmax_text=columns_list['xmax_text'][j]
            ymax_text=rows_list['ymax_text'][i]
            end_row=i
            start_col=j
            end_col=j
            
            # if (xmin_value,ymin_value) in spanningCell_list['xmin_ymin']:
            #     index=spanningCell_list['xmin_ymin'].index((xmin_value,ymin_value))
            #     xmax_value=spanningCell_list['xmax_ymax'][index][0]
            #     ymax_value=spanningCell_list['xmax_ymax'][index][1]
            #     end_row = rows_list['ymax'].index(ymax_value)
            #     end_col = columns_list['xmax'].index(xmax_value)
            #     xmax_text=spanningCell_list['text'][index]['xmax_val']
            #     ymax_text=spanningCell_list['text'][index]['ymax_val']
            #     if start_row!=end_row:
            #         dont_care[i+1:end_row+1,j]="True"
                    
            #     if start_col!=end_col:
            #         dont_care[i,j+1:end_col+1]="True"
    
            xml_out.add_cell(xmin_val=columns_list['xmin_text'][j],ymin_val=rows_list['ymin_text'][i],xmax_val=xmax_text,ymax_val=ymax_text,start_row=str(start_row),end_row=str(end_row),start_col=str(start_col),end_col=str(end_col),dont_care=dont_care[i][j])

    xml_out.save_xml(os.path.join(output_dir,xml_file))


with open('Files_With_Multiple_Tables_Test.txt','w') as file:
    for xml_file in files_with_multiple_tables:
        file.write(xml_file)
        file.write('\n')