"""
Example:
    python infer.py --config=configs/finetune_funsd.yaml
"""

import os
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from glob import glob
import cv2

from lightning_modules.data_modules.vie_dataset import VIEDataset
from model import get_model
from utils import get_class_names, get_config, get_label_map


cfg = get_config()
net = get_model(cfg)

net.eval()

"""
Required parameters: 
input_ids = batch["input_ids"]
image = batch["image"]
bbox = batch["bbox"]
bbox_4p_normalized = batch["bbox_4p_normalized"]
attention_mask = batch["attention_mask"]
first_token_idxes = batch["first_token_idxes"]
first_token_idxes_mask = batch["block_mask"]
line_rank_id = batch["line_rank_id"]
line_rank_inner_id = batch["line_rank_inner_id"]
"""

def main():
    mode = "val"
    cfg = get_config('configs/finetune_funsd.yaml')
    if cfg[mode].dump_dir is not None:
        cfg[mode].dump_dir = os.path.join(cfg[mode].dump_dir, cfg.workspace.strip('/').split('/')[-1])
    else:
        cfg[mode].dump_dir = ''
    print(cfg)

    if cfg.pretrained_model_file is None:
        pt_list = os.listdir(os.path.join(cfg.workspace, "checkpoints"))
        if len(pt_list) == 0:
            print("Checkpoint file is NOT FOUND!")
            exit(-1)
        pt_to_be_loaded = pt_list[0]
        if len(pt_list) > 1:
            # import ipdb;ipdb.set_trace()
            for pt in pt_list:
                if cfg[mode].pretrained_best_type in pt:
                    pt_to_be_loaded = pt
                    break
        cfg.pretrained_model_file = os.path.join(cfg.workspace, "checkpoints", pt_to_be_loaded)

    net = get_model(cfg)

    load_model_weight(net, cfg.pretrained_model_file)

    net.to("cuda")
    net.eval()

    if cfg.model.backbone in [
        "alibaba-damo/geolayoutlm-base-uncased",
        "alibaba-damo/geolayoutlm-large-uncased",
    ]:
        backbone_type = "geolayoutlm"
    else:
        raise ValueError(
            f"Not supported model: cfg.model.backbone={cfg.model.backbone}"
        )

    dataset = VIEDataset(
        cfg.dataset,
        cfg.task,
        backbone_type,
        cfg.model.head,
        cfg.dataset_root_path,
        net.tokenizer,
        mode=mode,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg[mode].batch_size,
        shuffle=False,
        num_workers=cfg[mode].num_workers,
        pin_memory=True,
        drop_last=False,
    )