
from merge import data_pipeline
import uuid
import shutil
import os, errno
from protogen import TableDetection_pb2
from protogen import TableDetection_pb2_grpc
import pdfbox
import cv2
from pathlib import Path
import glob
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from table_detection import make_prediction, plot_prediction
import torch
from termcolor import cprint
from libs.model import SplitModel
from libs.model import MergeModel
import libs.utils as utils
import numpy as np
from ast import literal_eval
import xmltodict as xml
from operator import itemgetter
import pandas as pd
import json
import base64
from Constants import Constants
from google.protobuf.json_format import Parse
import xml.etree.ElementTree as ET
from TableExtraction import TableExtraction
#from merge import MergeModel

class TableDetectorService(TableDetection_pb2_grpc.TableDetectionService):

    #create detectron config
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu'
    #set yaml
    cfg.merge_from_file('All_X152.yaml')

    #set model weights
    cfg.MODEL.WEIGHTS = 'model_final.pth' # Set path model .pth

    predictor = DefaultPredictor(cfg) 

    def CreateSession(self, request : TableDetection_pb2.SessionRequest, context):
        print('CreateSession request received : name : ', request.strImageName)
        strUniqueID = str(uuid.uuid4())
        strUinqueFolder = os.path.join(Constants.folderName, strUniqueID)
        try:
            os.makedirs(strUinqueFolder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        noOfPages = self.CreateOrUpdateFile(request.strImageName, strUinqueFolder, request.bFileContent)

        return TableDetection_pb2.SessionReply(bIsSuccess = True, strErrorMessage = '', strSessionID = strUniqueID, totalPages = noOfPages)

    def GetSession(self, request : TableDetection_pb2.GetSessionRequest, context):
        print('GetSession request received : name : ', request.strSessionID)
        strUniqueFolder = os.path.join(Constants.folderName, request.strSessionID)
        
        if(request.strPageNumber == '-1'):
            fileName = os.path.join(Constants.strImageFileName + '.*')
            detectedName = os.path.join(Constants.strDetectedTableCropFileSuffix + '.*')
            splitmergedName = os.path.join(Constants.strDetectedTableCropFileSuffix + '.*')
        else:
            fileName = os.path.join(Constants.strImageFileName + request.strPageNumber + '.*') 
            detectedName = os.path.join(Constants.strDetectedTableCropFileSuffix + request.strPageNumber + '.*')
            splitmergedName = os.path.join(Constants.strDetectedTableCropFileSuffix + request.strPageNumber + '.*')

        lstOrigFileContent = []
        lstDetectedFileContent = []
        lstSplitmergedFileContent = []
        fileCount = 0
        if(len(glob.glob1(strUniqueFolder,fileName)) > 0):
            for path in Path(strUniqueFolder).glob(fileName):
                with open(path, 'rb') as file:
                    lstOrigFileContent.append(file.read())
                print(path)
                fileCount+=1
        else:
            fileName = os.path.join(Constants.strImageFileName + '.*') 
            for path in Path(strUniqueFolder).glob(fileName):
                with open(path, 'rb') as file:
                    lstOrigFileContent.append(file.read())
                print(path)
                fileCount+=1

        fileCount = 0
        for path in Path(os.path.join(strUniqueFolder, "tabledetectedcropped")).glob(detectedName):
            with open(path, 'rb') as file:
                lstDetectedFileContent.append(file.read())
            print(path)
            fileCount+=1
        fileCount = 0
        for path in Path(os.path.join(strUniqueFolder, "output", "visualization")).glob(splitmergedName):
            with open(path, 'rb') as file:
                lstSplitmergedFileContent.append(file.read())
            print(path)
            fileCount+=1
        return TableDetection_pb2.GetSessionReply(bIsSuccess = True, strErrorMessage = '', lstFileContent = lstOrigFileContent, lstDetectedFileContent = lstDetectedFileContent, lstFinalFileContent = lstSplitmergedFileContent)

    def UpdateSession(self, request : TableDetection_pb2.UpdateRequest, context):
        print('request received : update : ', request.strImageName)
        strUinqueFolder = os.path.join(Constants.folderName, request.strSessionID)
        TableDetectorService.CreateOrUpdateFile(request.strImageName, strUinqueFolder, request.bFileContent)
        return TableDetection_pb2.SessionReply(bIsSuccess = True, strErrorMessage = '', strSessionID = '')

    def DeleteSession(self, request : TableDetection_pb2.SessionDeleteRequest, context):
        print('DeleteSession request received to delete folder : ', request.strSessionID)
        TableDetectorService.DeleteFolder(os.path.join(Constants.folderName , request.strSessionID))
        return TableDetection_pb2.SessionDeleteReply(bIsSuccess = True, strErrorMessage = '')

    def DetectTable(self, request : TableDetection_pb2.DetectTableRequest, context):
        print('DetectTable request received : name : ', request.strSessionID)
        # table detection
        strFileToDetect = os.path.join(Constants.folderName, request.strSessionID, Constants.strImageFileName + request.strPageNumber + Constants.strImageFileFormat)
        print('strFileToDetect' , strFileToDetect)
        if not os.path.isfile(strFileToDetect):
            strFileToDetect = os.path.join(Constants.folderName, request.strSessionID, Constants.strImageFileName + Constants.strImageFileFormat)
            print('strFileToDetect' , strFileToDetect)
        document_img = cv2.imread(strFileToDetect)
        plot_prediction(document_img, self.predictor, os.path.join(Constants.folderName, request.strSessionID), request.strPageNumber, Constants.strDetectedTableCropFileSuffix  + request.strPageNumber + Constants.strImageFileFormat)

        uniqueFolderPath = os.path.join(Constants.folderName, request.strSessionID) 
        image_path = os.path.join(Constants.folderName, request.strSessionID, Constants.strDetectedTableCropFolder)

        image_file = os.path.join(Constants.folderName, request.strSessionID, Constants.strDetectedTableCropFolder, Constants.strDetectedTableCropFileSuffix  + request.strPageNumber + Constants.strImageFileFormat)

        image_name = Constants.strDetectedTableCropFileSuffix  + request.strPageNumber + Constants.strImageFileFormat

        output_path = os.path.join(Constants.folderName, request.strSessionID, "output")
        model_weights = "TableDetection/model/split_model3999.pth"
        os.makedirs(output_path, exist_ok=True)
        if not os.path.isdir(os.path.join(output_path, "predicted_xmls")):
            os.makedirs(os.path.join(output_path, "predicted_xmls"))
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        cprint("creating split model...", "blue", attrs=["bold"])
        model = SplitModel(eval_mode=True).to(device)

        cprint("loading weights...", "blue", attrs=["bold"])
        model.load_state_dict(
            torch.load(model_weights, map_location=device)["model_state_dict"]
        )
        # to make predictions
        model.eval()

        cprint("Predicting table rows and columns:", "green", attrs=["bold"])
        print(40 * "-")
        
        xml_path = os.path.join(
                output_path, "predicted_xmls", image_name.split(".")[0] + ".xml"
            )
        
        rpn_out=[]
        cpn_out=[]
        with torch.no_grad():
            image = cv2.imread(image_file)
            H, W, C = image.shape
            image_trans = image.transpose((2, 0, 1)).astype("float32")
            resized_image = utils.resize_image(image_trans)
            input_image = utils.normalize_numpy_image(resized_image).unsqueeze(0)

            rpn_out, cpn_out = model(input_image.to(device))

            rpn_image = utils.probs_to_image(
                rpn_out.detach().clone(), input_image.shape, 1
            ).cpu()
            cpn_image = utils.probs_to_image(
                cpn_out.detach().clone(), input_image.shape, 0
            ).cpu()

            grid_img, row_image, col_image = utils.binary_grid_from_prob_images(
                rpn_image, cpn_image
            )
            
            grid_np_img = utils.tensor_to_numpy_image(grid_img)
            row_np_image = utils.tensor_to_numpy_image(row_image)
            col_np_image = utils.tensor_to_numpy_image(col_image)
        
            utils.process_output(image, row_np_image, col_np_image, xml_path)

            grid_np_img = cv2.resize(grid_np_img, (W, H))
            grid_np_img = cv2.cvtColor(grid_np_img, cv2.COLOR_GRAY2BGR)
            test_image = image.copy()
            test_image[np.where((grid_np_img == [255, 255, 255]).all(axis=2))] = [
                0,
                255,
                0,
            ]

            if not os.path.isdir(os.path.join(output_path, "images")):
                os.makedirs(os.path.join(output_path, "images"))
            cv2.imwrite(
                os.path.join(output_path, "images", image_name[:-4] + Constants.strImageFileFormat),
                test_image,
            )

            row_img = image.copy()
            rpn_image[rpn_image > 0.7] = 255
            rpn_image[rpn_image <= 0.7] = 0
            rpn_image = rpn_image.squeeze(0).squeeze(0).detach().numpy()
            rpn_image = cv2.resize(rpn_image, (W, H), interpolation=cv2.INTER_NEAREST)
            rpn_image = cv2.cvtColor(rpn_image, cv2.COLOR_GRAY2BGR)
            row_img[np.where((rpn_image == [255, 255, 255]).all(axis=2))] = [
                255,
                0,
                255,
            ]

            col_img = image.copy()
            cpn_image[cpn_image > 0.7] = 255
            cpn_image[cpn_image <= 0.7] = 0
            cpn_image = cpn_image.squeeze(0).squeeze(0).detach().numpy()
            cpn_image = cv2.resize(cpn_image, (W, H), interpolation=cv2.INTER_NEAREST)
            cpn_image = cv2.cvtColor(cpn_image, cv2.COLOR_GRAY2BGR)
            col_img[np.where((cpn_image == [255, 255, 255]).all(axis=2))] = [
                255,
                0,
                255,
            ]

        # Merge Model Prediction
        col_merge_list = []    
        row_merge_list = []    
        try:   
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            print("Creating merge model...")
            model = MergeModel().to(device)

            print("loading weights...")
            checkpoint = torch.load("/datadrive/code/src/Navanee_merge/Navanee/deep-splerge-dev/model_out/merge_model_3000.pth",map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])            
            model.eval()                  
            thresh =0.70
            row_prob = rpn_out[2].cpu()
            col_prob = cpn_out[2].cpu()

            image_shape = image.shape
            col_prob_img = utils.probs_to_image(col_prob.detach().clone(), image_shape, axis=0)
            row_prob_img = utils.probs_to_image(row_prob.detach().clone(), image_shape, axis=1)

            col_region = col_prob_img.detach().clone()
            col_region[col_region > thresh] = 1 
            col_region[col_region <= thresh] = 0
            col_region = (~col_region.bool()).float()

            row_region = row_prob_img.detach().clone()
            row_region[row_region > thresh] = 1
            row_region[row_region <= thresh] = 0
            row_region = (~row_region.bool()).float()    

            grid_img, row_img, col_img = utils.binary_grid_from_prob_images(row_prob_img, col_prob_img)

            # utils.tensor_to_numpy_image(row_img, write_path="../deep-splerge/eval/row_out/"+img_name+".png")
            # utils.tensor_to_numpy_image(col_img, write_path="../deep-splerge/eval/col_out/"+img_name+".png")   
            # continue         

            row_img = cv2.resize(row_img[0,0].numpy(), (W, H))
            col_img = cv2.resize(col_img[0,0].numpy(), (W, H))

            #gt_down, gt_right = utils.create_merge_gt(row_img, col_img, os.path.join(merges_path, img_name + ".pkl"))
            
            input_feature = torch.cat((image, 
                                    row_prob_img, 
                                    col_prob_img,
                                    row_region, 
                                    col_region, 
                                    grid_img), 
                                1)

            outputs = model(input_feature.to(device))

            row_merge = outputs[1].squeeze(0).squeeze(0)
            row_merge[row_merge > thresh] = 1
            row_merge[row_merge <= thresh] = 0
            
            col_merge = outputs[3].squeeze(0).squeeze(0)
            col_merge[col_merge > thresh] = 1
            col_merge[col_merge <= thresh] = 0

            # if configs.eval:
            #         row_tp += np.count_nonzero(((row_merge.cpu() == 1) & (gt_down == 1)).numpy())
            #         row_tn += np.count_nonzero(((row_merge.cpu() == 0) & (gt_down == 0)).numpy())
            #         row_fn += np.count_nonzero(((row_merge.cpu() == 0) & (gt_down == 1)).numpy())
            #         row_fp += np.count_nonzero(((row_merge.cpu() == 1) & (gt_down == 0)).numpy())

            #         col_tp += np.count_nonzero(((col_merge.cpu() == 1) & (gt_right == 1)).numpy())
            #         col_tn += np.count_nonzero(((col_merge.cpu() == 0) & (gt_right == 0)).numpy())
            #         col_fn += np.count_nonzero(((col_merge.cpu() == 0) & (gt_right == 1)).numpy())
            #         col_fp += np.count_nonzero(((col_merge.cpu() == 1) & (gt_right == 0)).numpy())

            grid_np_img = utils.tensor_to_numpy_image(grid_img)
            grid_np_img = cv2.resize(grid_np_img, (W,H))
            grid_np_img = cv2.cvtColor(grid_np_img, cv2.COLOR_GRAY2BGR)
            test_image = cv2.imread(image_file)
            test_image[np.where((grid_np_img == [255, 255, 255]).all(axis = 2))] = [0, 255, 0]

            out_img,row_list,col_list = utils.draw_merge_output(test_image, grid_img, col_merge, row_merge)
            #gt_img,row_list,col_list = utils.draw_merge_output(test_image, grid_img, gt_right, gt_down, colors=((0,100,255), (255,100,0)))
            out_img = cv2.copyMakeBorder(out_img, 0, 0, 10, 0, cv2.BORDER_CONSTANT)
            
            
            if(len(col_list) > 0):
                strFinalvalue =  col_list[0].split(',')[0]+","+col_list[0].split(',')[1]
                for j in range(0,len(col_list)):  
                        strvalue = col_list[j].split(',')                  
                        if(j == len(col_list)-1):
                            strFinalvalue = strFinalvalue+ ","+strvalue[2]+ ","+strvalue[3]#strvalue[0]+","+ strvalue[1]+","+col_list[j-1].split(',')[2]+","+col_list[j-1].split(',')[3]
                            col_merge_list.append(strFinalvalue)
                            break
                        if(float(strvalue[2]) != float(col_list[j+1].split(',')[0])):
                            strFinalvalue = strFinalvalue+ ","+strvalue[2]+ ","+strvalue[3]#strvalue[0]+","+ strvalue[1]+","+col_list[j-1].split(',')[2]+","+col_list[j-1].split(',')[3]
                            col_merge_list.append(strFinalvalue)
                            strFinalvalue = col_list[j+1].split(',')[0]+","+col_list[j+1].split(',')[1]
            if(len(row_list) > 0):
                strFinalvalue =  row_list[0].split(',')[0]+","+row_list[0].split(',')[1]
                for j in range(0,len(col_list)):  
                        strvalue = row_list[j].split(',')                  
                        if(j == len(col_list)-1):
                            strFinalvalue = strFinalvalue+ ","+strvalue[2]+ ","+strvalue[3]#strvalue[0]+","+ strvalue[1]+","+col_list[j-1].split(',')[2]+","+col_list[j-1].split(',')[3]
                            row_merge_list.append(strFinalvalue)
                            break
                        if(float(strvalue[3]) != float(row_list[j+1].split(',')[1])):
                            strFinalvalue = strFinalvalue+ ","+strvalue[2]+ ","+strvalue[3]#strvalue[0]+","+ strvalue[1]+","+col_list[j-1].split(',')[2]+","+col_list[j-1].split(',')[3]
                            row_merge_list.append(strFinalvalue)
                            strFinalvalue = row_list[j+1].split(',')[0]+","+row_list[j+1].split(',')[1]

            # compare = np.concatenate((gt_img, out_img), axis=1)
            # cv2.imwrite("outputs/"+img_name+".png", compare)
            # cv2.imshow("img", compare)
            # cv2.waitKey(0)
            # exit(0)
        except Exception as e:
            print(e)
        
        et = ET.parse(xml_path)

        rowMerge = row_merge_list
        colMerge = col_merge_list

        for strRowPoints in rowMerge:
            startCell = None
            endCell = None
            pointAMerge = strRowPoints.split(';')[0]
            pointBMerge = strRowPoints.split(';')[1]

            lineAX  = float(pointAMerge.split(',')[0])
            lineAY  = float(pointAMerge.split(',')[1])
            lineBX  = float(pointBMerge.split(',')[0])
            lineBY  = float(pointBMerge.split(',')[1])
            for name in et.getroot().iterfind('Tables/GroundTruth/Tables/Table/Cell'):
                if(TableDetectorService.contains(float(name.attrib['x0']) , float(name.attrib['y0']) , float(name.attrib['x1']) , float(name.attrib['y1']), lineAX, lineAY)):
                    startCell = name
                if(TableDetectorService.contains(float(name.attrib['x0']) , float(name.attrib['y0']) , float(name.attrib['x1']) , float(name.attrib['y1']), lineBX, lineBY)):
                    endCell = name

            if(startCell != None and endCell != None):
                print('updating endCol to ' + endCell.attrib['endCol'])
                startCell.attrib['endCol'] = endCell.attrib['endCol']
                endCell.attrib['dontCare'] = 'true'
            
        for strColPoints in colMerge:
            startCell = None
            endCell = None
            pointAMerge = strColPoints.split(';')[0]
            pointBMerge = strColPoints.split(';')[1]

            lineAX  = float(pointAMerge.split(',')[0])
            lineAY  = float(pointAMerge.split(',')[1])
            lineBX  = float(pointBMerge.split(',')[0])
            lineBY  = float(pointBMerge.split(',')[1])
            for name in et.getroot().iterfind('Tables/GroundTruth/Tables/Table/Cell'):
                
                if(TableDetectorService.contains(float(name.attrib['x0']) , float(name.attrib['y0']) , float(name.attrib['x1']) , float(name.attrib['y1']), lineAX, lineAY)):
                    startCell = name
                if(TableDetectorService.contains(float(name.attrib['x0']) , float(name.attrib['y0']) , float(name.attrib['x1']) , float(name.attrib['y1']), lineBX, lineBY)):
                    endCell = name
            if(startCell != None and endCell != None):
                startCell.attrib['endRow'] = endCell.attrib['endRow']
                endCell.attrib['dontCare'] = 'true'

        et.write(xml_path)
        # Open original file
        et = ET.parse(xml_path)

        # Append new tag: <a x='1' y='abc'>body text</a>
        new_tag = ET.SubElement(et.getroot(), 'a')
        new_tag.text = 'body text'
        new_tag.attrib['x'] = '1' # must be str; cannot be an int
        new_tag.attrib['y'] = 'abc'

        # Write back to file
        #et.write('file.xml')
        et.write(xml_path)
        # =================table ocr merge.py===================
        #os.system('python3 merge.py')

        os.makedirs(os.path.join(output_path, "xmls"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "visualization"), exist_ok=True)

        data_pipeline(
            os.path.join(output_path, "predicted_xmls", image_name.split(".")[0] + ".xml"), 
            output_path
            , os.path.join(output_path, "images", image_name[:-4] + Constants.strImageFileFormat),
            None)

        # print(tabdet)
        with open(os.path.join(output_path, "predicted_xmls", image_name.split(".")[0] + ".xml")) as xml_file:
        
            data_dict = xml.parse(xml_file.read())
            xml_file.close()
        
        # generate the object using json.dumps() 
        # corresponding to json data
        
        json_data = json.dumps(data_dict)
    
        
        # Write the json data to output 
        # json file

        if not os.path.isdir(os.path.join(output_path, "data")):
            os.makedirs(os.path.join(output_path, "data"))

        with open(os.path.join(output_path, "data", "data.json"), "w") as json_file:
            json_file.write(json_data)
            json_file.close()
    
        f = open(os.path.join(output_path, "data", "data.json"),)
    
        # returns JSON object as 
        # a dictionary  
        data = json.load(f)
        
        # Iterating through the json
        Rows=len(data['GroundTruth']['Tables']['GroundTruth']['Tables']['Table']['Row'])  
        Column=len(data['GroundTruth']['Tables']['GroundTruth']['Tables']['Table']['Column'])
        Cells=data['GroundTruth']['Tables']['GroundTruth']['Tables']['Table']['Cell']
        temp_cell=[]
        temp_row=[]
    
        final_array=[]
        row_obj={}
        text=""
        obj_arr=[]
        for i in range(0,Rows+1):
            count=0
            for cell in Cells:
                if cell["@startRow"]==str(i):
                    # print(cell["@startRow"])
                    x0=cell["@x0"]
                    x1=cell["@x1"]
                    y0=cell["@y0"]
                    y1=cell["@y1"]
                    with open(os.path.join(output_path, "bboxes", 'bboxes.txt')) as f:
                        for l in f:
                            x=literal_eval(l)
                            if (x[2] and x[4]  in range(int(x0),int(x1))) and  (x[3] and x[5] in range(int(y0),int(y1))):
                                temp_cell.append(x)
                        a = sorted(temp_cell, key=itemgetter(2))   
                        sorted_cell = sorted(a, key=itemgetter(3))  
                        temp_row.append(sorted_cell)
                        temp_cell=[]
            final_array.append(temp_row)
            temp_row=[]  

        self.cls()
    
        temp_dict={}
        count=0
        final_object=[]
        for i in final_array:
            for j in i:
                if j==[]:
                    temp_dict["col"+ str(count)]=" " #3 4
                if len(j)==1:
                    temp_dict["col"+ str(count)]=j[0][1] # 0 1 2
                if len(j)>1:
                    for k in j:
                        text=text+k[1]+" "
                    temp_dict["col"+ str(count)]=text
                    text=""
                count=count+1
            final_object.append(temp_dict)  
            print(temp_dict)  
            temp_dict={}
            count=0
        
        json_data = json.dumps(final_object)
        # Write the json data to output 
        # json file
        if not os.path.isdir(os.path.join(output_path, "output_json")):
            os.makedirs(os.path.join(output_path, "output_json"), exist_ok=True)
        with open(os.path.join(output_path, "output_json", "output.json"), "w") as json_file:
            json_file.write(json_data)
            json_file.close()
        # ==================response=====================
        with open(os.path.join(Constants.folderName, request.strSessionID, Constants.strDetectedTableCropFolder, Constants.strDetectedTableCropFileSuffix  + request.strPageNumber + Constants.strImageFileFormat), "rb") as img_file:
            tabdet = base64.b64encode(img_file.read())
        with open(os.path.join(Constants.folderName, request.strSessionID, Constants.strDetectedTableCropFolder, Constants.strDetectedTableCropFileSuffix  + request.strPageNumber + Constants.strImageFileFormat), "rb") as img_file:
            cell_det = base64.b64encode(img_file.read())
        # print(tabdet)
        raw_data = {'ocrdata':json_data}
        objDetectTableOutput = TableDetection_pb2.DetectTableOutput()
        objDetectTableOutput.objTableData = ''
        with open('test.txt', "w") as file:
            file.write(str(raw_data))
        return TableDetection_pb2.DetectTableReply(bIsSuccess = True, strErrorMessage = '', objDetectTableOutput = objDetectTableOutput)
        #return TableDetection_pb2.DetectTableReply(bIsSuccess = True, strErrorMessage = '')

    @staticmethod
    def cls():
        os.system('cls' if os.name=='nt' else 'clear')
    
    @staticmethod
    def CreateOrUpdateFile(strFileName, strUniqueFolder, bFileContent):
        TableDetectorService.DeleteFiles(strUniqueFolder)
        if strFileName.endswith(Constants.strPDFFileFormat):
            with open(os.path.join(strUniqueFolder, Constants.strImageFileName + Constants.strPDFFileFormat), 'wb') as file:
                file.write(bFileContent)
            pdf = pdfbox.PDFBox()
            pdf.pdf_to_images(os.path.join(strUniqueFolder, Constants.strImageFileName + Constants.strPDFFileFormat), imageType= 'png')
            print("pdf conversion")
        else:
            image_path=os.path.join(strUniqueFolder, 'input_temp' + Constants.strImageFileFormat)
            with open(image_path, 'wb') as file:
                file.write(bFileContent)
            img=cv2.imread(os.path.join(image_path))
            cv2.imwrite(os.path.join(strUniqueFolder,  Constants.strImageFileName + Constants.strImageFileFormat),img)

        print("image saved")
        return  len(os.listdir(strUniqueFolder))-1

    @staticmethod
    def DeleteFiles(strUniqueFolder):
        for filename in os.listdir(strUniqueFolder):
                file_path = os.path.join(strUniqueFolder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    @staticmethod
    def DeleteFolder(strUniqueFolder):
        try:
            if os.path.isdir(strUniqueFolder):
                shutil.rmtree(strUniqueFolder)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (strUniqueFolder, e))
    
    @staticmethod
    def contains(rx0, ry0, rx1, ry1, x, y):
        if(x >= rx0 and x <= rx1):
            if(y >= ry0 and y <= ry1):
                return True
        return False

    def ExtractTable(self, request : TableDetection_pb2.ExtractTableInput, context):
        objtab = TableExtraction()
        strImage = request.strImagearray
        strEngine = request.strEngine
        objdata = request.lstpages
        strUniqueID = str(uuid.uuid4())
        strUinqueFolder = os.path.join(Constants.folderName, strUniqueID)
        try:
            os.makedirs(strUinqueFolder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        noOfPages = self.CreateOrUpdateFile(request.strImageName, strUinqueFolder, strImage)

        output = objtab.ExtractTable(strUniqueID,1,None)
        return output