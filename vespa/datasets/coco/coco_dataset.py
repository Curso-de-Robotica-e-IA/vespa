import os
import json
import cv2
from pycocotools.coco import COCO
import torch
from vespa.datasets.base_dataset import BaseDataset


class COCODataset(BaseDataset):
    def __init__(self, root_dir, ann_file, transforms=None):
        super().__init__(root_dir, transforms)
        self.coco = COCO(ann_file)
        self.image_ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        for ann in annotations:
            xmin, ymin, width, height = ann['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann['category_id'])

        if self.transforms:
            augmented = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["labels"]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return img, target

    def __len__(self):
        return len(self.image_ids)
