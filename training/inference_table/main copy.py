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

def load_models_to_memory(int_cfg, run_module):
    loaded_models = {}
    device = int_cfg['MODEL']['DEVICE']
    device = device.lower().strip()
    if run_module == 'all' or run_module == 'table':
        detectron_model = int_cfg['MODEL']['DETECTION_MODEL']
        detectron_config = int_cfg['MODEL']['DETECTION_CONFIG']
        loaded_models['table'] = inference.detectron.load_model(
                            yaml=detectron_config,
                            model_weights= detectron_model,
                            device=device
                        )
    if run_module == 'all' or run_module == 'header':
        loaded_models['header'] = load_geo_model(device=device)
        
    if run_module == 'all' or run_module == 'row':
        model_args = load_model_config(int_cfg['MODEL']['TATR_CONFIG'])
        loaded_models['row'] = load_structure_model(model_args, device=device)
        
    return loaded_models

class Pipeline:
    def __init__(self, 
                document_image = None,
                table_image = None, 
                int_cfg = None, 
                ext_cfg = None, 
                image_name = None, 
                table_ocr = None,
                doc_ocr = None,
                mode = 'all',
                module = None,
                dump_path = 'data/output',
                loaded_models = {}):
        
        self.document_image = document_image
        if str(type(document_image)).endswith('NoneType>'):
            assert table_ocr != None
        elif str(type(table_image)).endswith('NoneType>'):
            assert not str(type(document_image)).endswith('NoneType>')
        self.table_image = table_image
        self.int_cfg = int_cfg
        self.ext_cfg = ext_cfg
        self.image_name = image_name
        self.loaded_models = loaded_models
        self.table_ocr = table_ocr
        self.data_dump_path = dump_path
        os.makedirs(self.data_dump_path, exist_ok=True)
        self.mode = mode
        self.module = module
        self.viz_dump_path = os.path.join(self.data_dump_path, 'final', 'table')
        os.makedirs(self.viz_dump_path, exist_ok=True)
        self.doc_ocr = doc_ocr
        
        os.makedirs(os.path.join(self.data_dump_path, 'ocr'), exist_ok=True)
        
        
        
    def table_detection(self, selected_ocr, ocr_client):
        table_res = {}
        padding = int(float(self.int_cfg['IMAGE']['CROP_PADDING']))
        try:
            table_list, table_coords = inference.detectron.run_detection(
                                        image=self.document_image,
                                        predictor = self.loaded_models['table'])
        except Exception as e:
            logging.error(e)
            return
        for i, table in enumerate(table_coords):
            table_coords = [table[0]-padding, table[1]-padding, table[2]+padding, table[3]+padding]
            self.table_image = self.document_image[table_coords[1]:table_coords[3], table_coords[0]:table_coords[2]]
            
            if not 0 in self.table_image.shape and  min(self.table_image.shape[:2]) > 100:
                document_name = self.image_name
                self.image_name = f"{self.image_name.split('.')[0]}_{i}.png"
                # #print(self.image_name)
                # #print(self.table_image.shape)
                table_image_path = os.path.join(self.data_dump_path, 'tables', self.image_name)
                cv2.imwrite(table_image_path, self.table_image)
                
                jsonname = self.image_name.replace('png','json')
                
                if selected_ocr != 'other':
                    self.table_ocr = utils.perform_ocr(ocr = selected_ocr, 
                                                       ocr_client = ocr_client, 
                                                       table_image_path = table_image_path)
                    # self.table_ocr = utils.clean_ocr(self.table_ocr)
                    with open(os.path.join(self.data_dump_path, 'ocr', jsonname), 'w') as f:
                        json.dump(self.table_ocr, f, indent=4)
                else:
                    self.table_ocr = utils.perform_ocr(ocr=selected_ocr,  
                                                       doc_ocr=self.doc_ocr,
                                                       table_coords = table_coords)
                    # self.table_ocr = utils.clean_ocr(self.table_ocr)
                    with open(os.path.join(self.data_dump_path, 'ocr', jsonname), 'w') as f:
                        json.dump(self.table_ocr, f, indent=4)
                table_res[self.image_name] = {
                    'table_coords': table
                }
                
                if self.module == 'all' and self.table_ocr != {}:
                
                    header_res = self.header_detection()
                    
                        
                    table_res[self.image_name]['header_strip'] = header_res['final_header']
                    
                    column_res = self.column_detection(header_res)
                
                    
                
                    row_res = self.row_detection(column_res, header_res['final_header'])
                
                    
                
                    self.visualize(table_coords, header_res, column_res, row_res)
                    
                        
        return table_res
            
    def header_detection(self):
        # #print("Starting header detection")
        header_res = detect_header(table_image=self.table_image,
                        int_cfg=self.int_cfg,
                        ext_cfg=self.ext_cfg,
                        table_image_name=self.image_name,
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
    
    def visualize(self, table, header, column, row):
        # if not str(type(self.document_image)).endswith('NoneType>'):
        header_elements = [item['header_cell_coordinate'] for item in column['meta']]
        header_coords = header['final_header']
        column_coords = column['column_seperators']
        # row_coords = [item['coords'] for item in row]
            
        # cv2.rectangle(self.table_image, (table[0], table[1]), (table[2], table[3]), (0, 0, 0), 2)
        if not header_coords == None:
            cv2.rectangle(self.table_image, (header_coords[0], header_coords[1]), (header_coords[2], header_coords[3]), (0, 255, 0), 2)
        
        for col in column_coords:
            col_coord = col[0]
            cv2.rectangle(self.table_image, (col_coord[0], col_coord[1]), (col_coord[2], col_coord[3]), (255, 0, 0), 2)
            
        for element in header_elements:
            cv2.rectangle(self.table_image, (element[0], element[1]), (element[2], element[3]), (0, 0, 255), 1)
            
            
        for row_item in row:
            approach = row_item['approach']
            row_coord = row_item['coords']
            
            if approach == 'tatr':
                line_color = (255, 0, 0)
            else:
                line_color = (0, 0, 255)
            
            cv2.rectangle(self.table_image, (row_coord[0], row_coord[1]), (row_coord[2], row_coord[3]), line_color, 2)
            
        
        cv2.imwrite(os.path.join(self.viz_dump_path, self.image_name), self.table_image)
# def pipeline(document_image=None, table_image=None):
#     pass    


# def header_detection(cropped_table, int_cfg, ext_cfg, table_image_name, table_ocr, loaded_models):
    

def main():
    
    
    
    logs_path = 'log'
    log_name = f"{str(datetime.now())}.txt"
        
    logging.basicConfig(filename=os.path.join(logs_path, log_name), 
                        filemode='w', 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    track_time = {}
    
    ext_cfg = utils.load_config('config.ini')
    int_cfg = utils.load_config('utils/internal_config.ini')
    batch_mode = ext_cfg['MODE']['BATCH']
    if batch_mode.lower().strip() == 'true':
        run_module = None
        section_modules = ext_cfg['MODULE']
        for item in section_modules:
            if section_modules[item].lower().strip() == 'true':
                run_module = item
        if run_module == None:
            exit("Activate one of the module to be run in batch mode")
    else:
        run_module = 'all'
    

    # geo_model = ext_cfg['MODEL']['GEOlAYOUT']
    
    dataset_name = ext_cfg['DATASET']['NAME']

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

    if run_module == 'table' or run_module == 'all':
        images_list = os.listdir(doc_input_path)
    else:
        images_list = os.listdir(table_dump_loc)

    
    
    
    track_time['model'] = {}
    track_time['model']['start_time'] = time()
    
    logging.info(f"Model loading onto {int_cfg['MODEL']['DEVICE']}")
    
    loaded_models = load_models_to_memory(int_cfg, run_module)
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
        if run_module == 'all' or run_module == 'table':
            image = cv2.imread(os.path.join(doc_input_path, filename))
            doc_ocr = None
            if selected_ocr == 'other' : 
                with open(os.path.join(ocr_dump, filename.replace('png','json')), 'r') as f:
                    doc_ocr = json.load(f)
            pipeline = Pipeline(document_image = image,
                                int_cfg = int_cfg,
                                ext_cfg = ext_cfg,
                                dump_path = data_dump_path,
                                mode = batch_mode,
                                module = run_module,
                                image_name=filename,
                                loaded_models=loaded_models,
                                doc_ocr=doc_ocr)
        else:
            table_name = filename
            image = cv2.imread(os.path.join(table_dump_loc, table_name))
            with open(os.path.join(ocr_dump, table_name.replace('png','json')), 'r') as f:
                table_ocr = json.load(f)
            pipeline = Pipeline(table_image = image, 
                                int_cfg = int_cfg, 
                                ext_cfg = ext_cfg, 
                                image_name = table_name, 
                                table_ocr = table_ocr,
                                dump_path = data_dump_path,
                                mode = batch_mode,
                                module = run_module,
                                loaded_models=loaded_models)
        
        if run_module == 'all' or run_module == 'table':
            table_res = pipeline.table_detection(selected_ocr=selected_ocr,
                                     ocr_client=ocr_client)
        if run_module == 'header':
            
            header_res = pipeline.header_detection()
            # #print(header_res)
        
            with open(os.path.join(header_json_dump, table_name.replace('png','json')), 'w') as f:
                json.dump(header_res, f, indent=4)
            final_header = header_res['final_header']
            if not final_header == None:
                cv2.rectangle(image, 
                            (final_header[0], final_header[1]),
                            (final_header[2], final_header[3]),
                            (255, 0, 0),
                            1)
            for item in table_ocr:  
                bbox = table_ocr[item]['bbox']
                cv2.rectangle(image, 
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            (0, 0, 0),
                            1)
            cv2.imwrite(os.path.join(header_dump_path, table_name), image)
            
        track_time['extraction'][f"image_{i}"]['end_time'] = time()
        track_time['extraction'][f"image_{i}"]['total'] = track_time['extraction'][f"image_{i}"]['end_time'] - track_time['extraction'][f"image_{i}"]['start_time']
        #print('+'*20)
        #print(track_time['extraction'][f"image_{i}"])
        #print('+'*20)
        i += 1
        
    log_name = str(datetime.now())
    with open(os.path.join('log','time',f'{log_name}.json'), 'w') as f:
        json.dump(track_time, f, indent=4)
    # log_file.close()
        # #print(filename)
        

        # for i, table in enumerate(table_coords):
        #     cropped_table = img[table[1]-padding:table[3]+padding, table[0]-padding:table[2]+padding]

        #     if not 0 in cropped_table.shape:
                
        #         table_image_path = os.path.join(table_dump_loc, table_image_name)
                
        #         cv2.imwrite(table_image_path, cropped_table)

                
        #         #utils.extract_text(ocr_client, table_image_path)
                
        #         jsonname = table_image_name.replace('png','json')
        #         with open(os.path.join(ocr_dump, jsonname), 'w') as f:
        #             json.dump(table_ocr, f, indent=4)
                    
                    
        #         if run_module == 'all' or run_module == 'header': 
                    
        #             res = header_detection()
        #             #print(res)
        #         exit()
            # geo_out = inference.geo.run_prediction(geo_config, geo_model, table_image_path, table_ocr)
            # #print(geo_out)
            # exit()
            
            
            # dict_header = header.generate.get()

if __name__ == '__main__':
    main()