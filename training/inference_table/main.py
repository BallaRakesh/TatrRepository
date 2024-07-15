import os
import utils
import inference
from columns.run import detect_columns
from rows.run import detect_rows
from PIL import Image
import cv2
import json
from header import load_geo_model
from header.run import detect_header
from tqdm import tqdm    
import numpy as np
from inference.tatr import load_structure_model, load_model_config
from time import time
from datetime import datetime
import logging
import psutil
import warnings
import copy

warnings.filterwarnings('ignore')

def load_models_to_memory(int_cfg):
    loaded_models = {}
    device = int_cfg['MODEL']['DEVICE']
    device = device.lower().strip()
    detectron_model = int_cfg['MODEL']['DETECTION_MODEL']
    detectron_config = int_cfg['MODEL']['DETECTION_CONFIG']
    loaded_models['table'] = inference.detectron.load_model(
                        yaml=detectron_config,
                        model_weights= detectron_model,
                        device=device
                    )
    loaded_models['header'] = load_geo_model(device=device, config=int_cfg['MODEL']['GEOlAYOUT_CONFIG'])
    
    model_args = load_model_config(int_cfg['MODEL']['TATR_CONFIG'])
    loaded_models['row'] = load_structure_model(model_args, device=device)
        
    return loaded_models

class Pipeline:
    def __init__(self, 
                int_cfg = None, 
                ext_cfg = None, 
                image_name = None, 
                table_ocr = None,
                doc_ocr = None,
                mode = 'all',
                module = None,
                dump_path = 'data/output',
                loaded_models = {}):
        
        # self.document_image = document_image
        # if str(type(document_image)).endswith('NoneType>'):
        #     assert table_ocr != None
        self.int_cfg = int_cfg
        self.ext_cfg = ext_cfg
        self.image_name = image_name
        self.loaded_models = loaded_models
        self.table_ocr = table_ocr
        self.data_dump_path = dump_path
        os.makedirs(self.data_dump_path, exist_ok=True)
        self.viz_table_dump_path = os.path.join(self.data_dump_path, 'final', 'table')
        os.makedirs(self.viz_table_dump_path, exist_ok=True)

        self.viz_doc_dump_path = os.path.join(self.data_dump_path, 'final', 'doc')
        os.makedirs(self.viz_doc_dump_path, exist_ok=True)
        self.doc_ocr = doc_ocr
        
        os.makedirs(os.path.join(self.data_dump_path, 'ocr'), exist_ok=True)
        
        
    
        
    def table_extraction(self, document_image):
        # # self.document_image = document_image
        # # self.table_image = document_image
        # # self.table_ocr = self.doc_ocr

        # # header_res = self.header_detection()
        # # column_res = self.column_detection(header_res)
        # #     # print(column_res['column_seperators'])
        # # row_res = self.row_detection(column_res,header_res['final_header'])
        
        # # self.visualize(header= header_res, column= column_res, row= row_res)
        self.document_image = document_image
        
        table_res = {}
        padding = int(float(self.int_cfg['IMAGE']['CROP_PADDING']))
        
        table_data = self.table_detection(document_image)

        if table_data != None:

            logging.info(f"\nTable Confidence: {table_data['conf']}\n")

            table = table_data['table_coords']
            table = [table[0] , table[1], table[2], table[3]]
            table_res[self.image_name] = {
                    'table_coords': table
                }
            
            jsonname = self.image_name.replace('png','json')
            with open(os.path.join(self.data_dump_path, 'ocr', jsonname), 'w') as f:
                json.dump(self.table_ocr, f, indent=4)

            if table_data['header']['final_header'] != [] and table_data['header']['final_header'] != None:
                        
                new_table_ymin = table_data['header']['final_header'][1]

                y_diff = new_table_ymin

                self.table_ocr = self.adjust_table(table_data['header']['final_header'], table_data['table_ocr'], y_diff)

                table = [table[0], table[1]+new_table_ymin, table[2], table[3]]
                table_res[self.image_name]['table_coords'] = table

                
                self.table_image = self.document_image[table[1]:table[3],
                                                        table[0]:table[2]]

                table_data['header']['final_header'][1] -= y_diff
                table_data['header']['final_header'][3] -= y_diff
                
            # print(table_data['header']['element_words'])
            table_res[self.image_name]['header_strip'] = table_data['header']['final_header']

            
            column_res = self.column_detection(table_data['header'])
            # print(column_res['column_seperators'])
            row_res = self.row_detection(column_res, table_data['header']['final_header'])
            self.visualize(table, table_data['header'], column_res, row_res)
            

            self.visualize(table=table, header=table_data['header'])
        else:
            self.visualize()
            return []

        # if table_list == []:
        #     self.visualize()
        #     return []

        # for table_id, table in enumerate(table_coords):
            
        #     # logging.info(f'Table Coordinate: {table}')
            
            
        #     # logging.info(f"ASPECT RATIO = {self.table_image.shape[1]/self.table_image.shape[0]}")
        #     # print(table_coords)
        #     # aspect_ratio = self.table_image.shape[1]/self.table_image.shape[0] # width to height
        #     # print(aspect_ratio)
        #     if table[2] - table[0] > 50:
        #         # document_name = self.image_name
                
                
        #         self.table_ocr, table = utils.perform_ocr(ocr='other',  
        #                                                 doc_ocr=self.doc_ocr,
        #                                                 table_coords = table)
        #             # self.table_ocr = utils.clean_ocr(self.table_ocr)
                    
                

        #         # updated_table_y2 = utils.adjust_table_coordinates(self.table_ocr, table)
        #         # table = [max(0, table[0]-padding), max(0, table[1]-padding), max(0, table[2]+padding), max(0, updated_table_y2+padding)]

        #         if table[3] - table[1] <= padding*2:
        #             self.visualize()
        #             return []

        #         self.table_image = document_image[table[1]:table[3], table[0]:table[2]]
        #         self.image_name = f"{self.image_name.split('.')[0]}_{table_id}.png"
        #         # #print(self.image_name)
        #         # #print(self.table_image.shape)
        #         table_image_path = os.path.join(self.data_dump_path, 'tables', self.image_name)
        #         cv2.imwrite(table_image_path, self.table_image)


        #         jsonname = self.image_name.replace('png','json')
        #         with open(os.path.join(self.data_dump_path, 'ocr', jsonname), 'w') as f:
        #             json.dump(self.table_ocr, f, indent=4)
                
                
        #         table_res[self.image_name] = {
        #             'table_coords': table
        #         }
                
        #         if self.table_ocr != {}:
                
        #             header_res = self.header_detection()
                    

            # # #         if header_res['final_header'] != [] and header_res['final_header'] != None:
                        
            # # #             new_table_ymin = header_res['final_header'][1]

            # # #             y_diff = new_table_ymin

            # # #             self.table_ocr = self.adjust_table(header_res['final_header'], self.table_ocr, y_diff)

            # # #             table = [table[0], table[1]+new_table_ymin, table[2], table[3]]
            # # #             table_res[self.image_name]['table_coords'] = table

            # # #             self.table_image = self.document_image[table[1]:table[3],
            # # #                                                    table[0]:table[2]]

            # # #             header_res['final_header'][1] -= y_diff
            # # #             header_res['final_header'][3] -= y_diff
                        
            # # #         # print(header_res['element_words'])
            # # #         table_res[self.image_name]['header_strip'] = header_res['final_header']

            # # #         # # if header_res['final_header'] != [] and header_res['final_header'] != None:
            # # #         # #     # print(table)
            # # #         # #     # print(header_res['final_header'])
            # # #         # #     new_table_start = [table[0] + header_res['final_header'][0] , table[1] + header_res['final_header'][1]]
            # # #         # #     table = new_table_start + table[2:]
            # # #         # #     # self.table_ocr = utils.remove_words(table, self.table_ocr)
            # # #         # #     table_res[self.image_name]['table_coords'] = table

            # # #         column_res = self.column_detection(header_res)
            # # #         # print(column_res['column_seperators'])
            # # #         row_res = self.row_detection(column_res, header_res['final_header'])
                    
            # # #         self.visualize(table, header_res, column_res, row_res)
            # # #     else:
            # # #         self.visualize()
            # # #         logging.warning('Empty OCR, table extraction terminated')
                                
            # # #     return table_res
            # # # else:
            # # #     self.visualize(table=table)
            # # #     logging.warning("No table detected")

    def adjust_table(self, header_bbox, table_ocr, y_diff):
        new_table_ocr = {}
        for word in table_ocr :
            if table_ocr[word]['vertices'][0][1] >= header_bbox[1]:
                new_table_ocr[word] = table_ocr[word]
                bbox = table_ocr[word]['bbox']
                bbox = [bbox[0], bbox[1]-y_diff, bbox[2], bbox[3]-y_diff]

                new_table_ocr[word]['bbox'] = bbox
                new_table_ocr[word]['vertices'] = [
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[1]),
                    (bbox[2], bbox[3]),
                    (bbox[0], bbox[3])
                ]
                
        del table_ocr
        return new_table_ocr
    

    def validate_table(self, table_coords, document_image):
        
        
        self.table_ocr, table_coords = utils.perform_ocr(ocr='other',  
                                                    doc_ocr=self.doc_ocr,
                                                    table_coords = table_coords)
        self.table_image = document_image[table_coords[1]:table_coords[3], table_coords[0]:table_coords[2]]
        header_res = self.header_detection()

        res = {'is_table':False, 
               'header': header_res, 
               'masked_image':None, 
               'n_masked': 0,
               'table_ocr':self.table_ocr, 
               'table_coords':table_coords}
        
        if header_res['final_header'] != None and header_res['final_header'] != []:
            header_text_width = header_res['header_text_region'][2] - header_res['header_text_region'][0]
            header_width = header_res['final_header'][2] - header_res['final_header'][0]
            logging.info(f"Header text region threshold: {header_text_width/(header_width)}")
            if len(header_res['element_coordinate']) <= 2:
                document_copy = copy.deepcopy(document_image)
                document_copy[0:table_coords[3], document_image.shape[1]:table_coords[2]] = 255
                res['masked_image'] = document_copy
                res['n_masked'] += 1
                # self.table_detection(document_image=document_copy)
            else:
                res['is_table'] = True
                res['header'] = header_res
                res['table_ocr'] = self.table_ocr
                res['table_coords'] = table_coords

        return res


    def table_detection(self, document_image):
        
        # try:
        table_data = inference.detectron.run_detection(
                                    image=document_image,
                                    predictor = self.loaded_models['table'])
        
        logging.info(table_data)
        # print(table_data)
        # confs = [table['conf'] for table in table_data['tables']]
        if table_data != {}:
            for table in table_data:
                if table['conf'] > float(self.int_cfg['MODEL']['TABLE_MIN_CONFIDENCE']):
                    table_meta = self.validate_table(table_coords=table['coords'],
                                                        document_image=document_image)
                    # return table_meta
                    if table_meta['is_table']:
                        table_meta['conf'] = table['conf']
                        return table_meta
                    elif table_meta['n_masked'] > 1:
                        self.visualize()
                        
                        # if not table_meta['n_masked'] > 1 and not str(type(table_meta['masked_image'])).endswith("'NoneType'>"):
                        #     # self.visualize(table=table_meta['table_coords'])
                        #     self.table_detection(table_meta['masked_image'])
                            
                else:
                    table_meta = self.validate_table(table_coords=table['coords'],
                                                        document_image=document_image)
                    # return table_meta
                    if table_meta['is_table']:
                        table_meta['conf'] = table['conf']
                        return table_meta
                    elif table_meta['n_masked'] > 1:
                        self.visualize()
                        # else:
                        #     if not table_meta['n_masked'] > 1 and not str(type(table_meta['masked_image'])).endswith("'NoneType'>"):
                        #         # self.visualize(table=table_meta['table_coords'])
                        #         self.table_detection(table_meta['masked_image'])
        # except Exception as e:
        #     logging.error(e)
        #     exit(e)
        #     return
        
        # return table_coords
    
    
    def header_detection(self):
        # #print("Starting header detection")
        header_res = detect_header(table_image=self.table_image,
                        int_cfg=self.int_cfg,
                        ext_cfg=self.ext_cfg,
                        ocr_data=self.table_ocr,
                        geo_model=self.loaded_models['header'])
            
        
        return header_res 
    
    def column_detection(self,header_res):
        single_line_elements_path = self.int_cfg['STATIC']['SINGLE_LINE_ELEM_PATH']
        with open(single_line_elements_path, 'r') as f:
            single_line_elements = json.load(f)
        single_line_elements = single_line_elements['headers']
        column_res = detect_columns(header_res=header_res,
                                    image=self.table_image,
                                    ocr_data=self.table_ocr,
                                    single_line_elements=single_line_elements)
        return column_res
    
    def row_detection(self, column_res, header_strip):

        # # rows = utils.get_lines(self.table_image)

        # # final_rows = []
        # # for r_id in rows:
        # #     if header_strip != None:
        # #         if rows[r_id]['block_bbox'][1] == header_strip[3]:
        # #             final_rows.append(rows[r_id]['block_bbox'])
        # #         # if rows[r_id]['block_bbox'][1] >= header_strip[3] :
        # #             final_rows.append(rows[r_id]['block_bbox'])
        # #     else:
        # #         final_rows.append(rows[r_id]['block_bbox'])
        # # return final_rows
        model_args = load_model_config(self.int_cfg['MODEL']['TATR_CONFIG'])
        # #print(f"HEADER STRIP: {header_strip}")
        # #print('\n\n\n')
        rows = detect_rows(
            model_args= model_args,
            model= self.loaded_models['row'],
            image = self.table_image,
            annotations=self.table_ocr,
            column_data= column_res,
            header_data=header_strip
        )
        
        return list(rows.values())
    
    def visualize(self, table=None, header=None, column=None, row=None):
        # if not str(type(self.document_image)).endswith('NoneType>'):

        header_elements = None
        column_coords = None

        if column != None:
            header_elements = [item['header_cell_coordinate'] for item in column['meta']]
            column_coords = column['column_seperators']
        
        header_coords = None
        if header != None:
            header_coords = header['final_header']
        
        # row_coords = [item['coords'] for item in row]
            
        # cv2.rectangle(self.table_image, (table[0], table[1]), (table[2], table[3]), (0, 0, 0), 2)
        if not header_coords == None and not header_coords == []:
            cv2.rectangle(self.table_image, (header_coords[0], header_coords[1]+2), (header_coords[2], header_coords[3]), (0, 255, 0), 4)

        # if header != None:
        #     header_coords = header['dict_coordinate']
        
        # # row_coords = [item['coords'] for item in row]
            
        # # cv2.rectangle(self.table_image, (table[0], table[1]), (table[2], table[3]), (0, 0, 0), 2)
        # if not header_coords == None and not header_coords == []:
        #     cv2.rectangle(self.table_image, (header_coords[0], header_coords[1]+2), (header_coords[2], header_coords[3]), (0, 255, 255), 4)
        
        if column_coords != [] and column_coords != None:
            for col in column_coords:
                col_coord = col[0]
                cv2.rectangle(self.table_image, (col_coord[0], col_coord[1]), (col_coord[2], col_coord[3]), (255, 0, 0), 2)
        if header_elements != []  and header_elements != None:
            for element in header_elements:
                cv2.rectangle(self.table_image, (element[0], element[1]), (element[2], element[3]), (0, 0, 255), 1)
            
        if row != [] and row != None:
            for row_item in row:
                # approach = ''
                approach = row_item['approach']
                row_coord = row_item['coords']
                
                if approach == 'tatr':
                    line_color = (255, 0, 0)
                else:
                    line_color = (0, 0, 255)
                
                cv2.rectangle(self.table_image, (row_coord[0], row_coord[3]), (row_coord[2], row_coord[3]), line_color, 2)

        if table != [] and table != None:

            self.document_image[table[1]:table[3], table[0]:table[2]] = self.table_image

            cv2.rectangle(self.document_image, (table[0], table[1]), (table[2], table[3]), (244, 0, 0), 3)

            cv2.imwrite(os.path.join(self.viz_table_dump_path, self.image_name), self.table_image)

        
        
        cv2.imwrite(os.path.join(self.viz_doc_dump_path, self.image_name), self.document_image)

def main():
    
    logs_path = 'log'
    log_name = f"{str(datetime.now())}.log"
        
    logging.basicConfig(filename=os.path.join(logs_path, log_name), 
                        filemode='a', 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s -- {%(pathname)s:%(lineno)d}',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=logging.DEBUG,
                        force=True)
    logging.info("Running Table Extraction")
    
    pid = os.getpid()
    python_process = psutil.Process(pid)
    memoryUse = python_process.memory_info()[0]/2.**30 
    logging.info(f"CPU consumption: {memoryUse} GB")
    
    track_time = {}
    
    ext_cfg = utils.load_config('config.ini')
    int_cfg = utils.load_config('utils/internal_config.ini')
    
    dataset_name = ext_cfg['DATASET']['NAME']
    
    logging.info(f'Dataset Name: {dataset_name}')
    
    doc_input_path = ext_cfg['PATH']['DOC_PATH']
    root_dump_path = ext_cfg['PATH']['DUMP_PATH']
    
    os.makedirs(root_dump_path, exist_ok=True)
    data_dump_path = os.path.join(root_dump_path, dataset_name)
    
    

    table_dump_loc = os.path.join(data_dump_path, 'tables')
    col_dump_loc = os.path.join(data_dump_path, 'columns')
    row_dump_loc = os.path.join(data_dump_path, 'rows')
    header_dump_loc = os.path.join(data_dump_path, 'headers')
    
    os.makedirs(table_dump_loc, exist_ok=True)
    os.makedirs(col_dump_loc, exist_ok=True)
    os.makedirs(row_dump_loc, exist_ok=True)
    os.makedirs(header_dump_loc, exist_ok=True)
    
    
    selected_ocr = 'tesseract'  # default
    for item in ext_cfg['OCR']:
        if ext_cfg['OCR'][item].lower() == 'true':
            selected_ocr = item
            break
    # #print(selected_ocr)
    # exit()
    

    credentials_path = ext_cfg['PATH']['GV_CREDENTIALS']

    if selected_ocr != 'other':
        ocr_dump = os.path.join(data_dump_path, 'ocr')
        
        os.makedirs(ocr_dump, exist_ok=True)
    else:
        ocr_dump = ext_cfg['PATH']['OCR_PATH']

    images_list = os.listdir(doc_input_path)

    
    
    
    track_time['model'] = {}
    track_time['model']['start_time'] = time()
    
    logging.info(f"Model loading onto {int_cfg['MODEL']['DEVICE']}")
    
    loaded_models = load_models_to_memory(int_cfg)
    
    memoryUse = python_process.memory_info()[0]/2.**30 
    logging.info(f"CPU consumption: {memoryUse} GB")
    
    track_time['model']['end_time'] = time()
    track_time['model']['total_time'] = track_time['model']['end_time'] - track_time['model']['start_time']
    
    logging.info(f"Model loaded in {track_time['model']['total_time']}s")
    # #print('====================================================================')
    # #print(track_time['model'])
    # #print("====================================================================")
    ocr_client = utils.get_ocr_client(selected_ocr, credentials_path)
    # geo_config, geo_model = inference.geo.load_geo_model()

    header_dump_path = os.path.join(data_dump_path, 'headers','final')
    header_json_dump = os.path.join(data_dump_path, 'headers','final_json')
    os.makedirs(header_json_dump, exist_ok=True)
    os.makedirs(header_dump_path, exist_ok=True)
    
    
    i = 0
    track_time['extraction'] = {}
    for filename in tqdm(images_list, 'Processing table extraction'):
        logging.info(f"Processing: {filename}")
        track_time['extraction'][f"image_{i}"] = {
            'image_name' : filename,
            'start_time' : time()
        }
        image = cv2.imread(os.path.join(doc_input_path, filename))
        doc_ocr = None
        if selected_ocr == 'other' : 
            with open(os.path.join(ocr_dump, filename.replace('png','json')), 'r') as f:
                doc_ocr = json.load(f)
        
        pipeline = Pipeline(int_cfg = int_cfg,
                            ext_cfg = ext_cfg,
                            dump_path = data_dump_path,
                            image_name=filename,
                            loaded_models=loaded_models,
                            doc_ocr=doc_ocr)
    
        table_res = pipeline.table_extraction(document_image=image)
        
        memoryUse = python_process.memory_info()[0]/2.**30 
        logging.info(f"CPU consumption: {memoryUse} GB")
        
        track_time['extraction'][f"image_{i}"]['end_time'] = time()
        track_time['extraction'][f"image_{i}"]['total'] = track_time['extraction'][f"image_{i}"]['end_time'] - track_time['extraction'][f"image_{i}"]['start_time']
        # #print('+'*20)
        # #print(track_time['extraction'][f"image_{i}"])
        # #print('+'*20)
        i += 1
        
    log_name = str(datetime.now())
    with open(os.path.join('log','time',f'{log_name}.json'), 'w') as f:
        json.dump(track_time, f, indent=4)

if __name__ == '__main__':
    main()