import os
import cv2
import inference.table_detection as table_detection
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from numpy import ndarray
from typing import List, Tuple


def load_model(yaml, model_weights, device):
    # Create Detectron2 configuration
    cfg = get_cfg()
    cfg.merge_from_file(yaml)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = device
    # Load the model
    predictor = DefaultPredictor(cfg)
    return predictor

def run_detection(image: ndarray, 
                   predictor: object) -> Tuple[List[ndarray], List]:
    """
    Run a table detection prediction using Detectron2.

    Loads a model based on the provided YAML configuration and weight files,
    performs prediction on the given image, and optionally saves the detected
    tables to the specified dump path.

    Parameters:
    image (ndarray): The input image on which table detection is to be performed.
    yaml (str): Path to the YAML configuration file for the Detectron2 model.
    weights (str): Path to the weights file (.pth) for the Detectron2 model.
    filename (str): Base filename to use for saving detected tables.
    dump_path (str): Directory path to save the detected tables. If empty, no files are saved.

    Returns:
    dict: A dicitonary containing table info, including:
           - table coordinate
           - confidence scores
    """

    
#     #print("Running Table Detection")

    # Perform the prediction
    tables = table_detection.make_prediction(image, predictor)
    
    # Save the detected tables if a dump path is provided
    # if dump_path:
    #     for t_no, table in enumerate(table_list):
    #         cv2.imwrite(os.path.join(dump_path, f"{filename}_{t_no}.png"), table)
            
    return tables
