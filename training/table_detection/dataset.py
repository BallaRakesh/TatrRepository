from detectron2.data.datasets import register_coco_instances
register_coco_instances("train", {}, "Multi-Type-TD-TSR/dataset/train/coco.json", "Multi-Type-TD-TSR/dataset/train/train_img")
register_coco_instances("test", {}, "Multi-Type-TD-TSR/dataset/val/coco.json", "Multi-Type-TD-TSR/dataset/val/val_img")
