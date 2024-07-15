import torch
import cv2
from detectron2.data.catalog import DatasetCatalog, Metadata, MetadataCatalog
from detectron2.engine import DefaultTrainer, TorchProfiler
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
import os
import random
from detectron2.utils.visualizer import Visualizer


from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_train", {}, "table_detection/dataset_final/train/train.json", "table_detection/dataset_final/train/images")
register_coco_instances("my_test", {}, "table_detection/dataset_final/test/test.json", "table_detection/dataset_final/test/images")


""" Metadata(evaluator_type='coco', image_root='Multi-Type-TD-TSR/dataset/train/train_img', json_file='Multi-Type-TD-TSR/dataset/train/coco.json', name='my_train',
         thing_classes=['table'], thing_dataset_id_to_contiguous_id={1: 0})
fruits_nuts_metadata = MetadataCatalog.get("my_train")

dataset_dicts = DatasetCatalog.get("my_train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    #cv2.imshow(vis.get_image()[:, :, ::-1]) """

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        coco_evaluator = COCOEvaluator(dataset_name, output_dir=output_folder)
        
        evaluator_list = [coco_evaluator]
        
        return DatasetEvaluators(evaluator_list)

cfg = get_cfg()
cfg.merge_from_file("table_detection/All_X152.yaml") #Get the basic model configuration from the model zoo 
#cfg.MODEL.DEVICE = 'cpu'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CustomTrainer(cfg) #previously DefaultTrainer(cfg    - because evaluator was not previously present, include coco evaluator too
trainer.resume_or_load(resume=True)
#add memory profiler
#TorchProfiler(lambda trainer: 10 < trainer.iter < 20, cfg.OUTPUT_DIR)
trainer.train()
print("Model Dump Path : ",os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
