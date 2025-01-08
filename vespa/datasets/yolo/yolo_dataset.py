import os
import cv2
import numpy as np
import torch
from vespa.datasets.base_dataset import BaseDataset


class YOLODataset(BaseDataset):
    def __init__(self, root_dir, txt_file, image_size, transforms=None):
        super().__init__(root_dir, transforms)
        self.image_size = image_size

        # Lê o arquivo com a lista de imagens
        with open(os.path.join(root_dir, txt_file)) as f:
            self.images = f.read().strip().split("\n")

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = img_path.replace('\images', '\labels').replace('.jpg', '.txt')

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Rótulo não encontrado: {label_path}")

        boxes = []
        labels = []

        with open(label_path) as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                labels.append(int(class_id))

                xmin = (x_center - width / 2) * img.shape[1]
                ymin = (y_center - height / 2) * img.shape[0]
                xmax = (x_center + width / 2) * img.shape[1]
                ymax = (y_center + height / 2) * img.shape[0]
                boxes.append([xmin, ymin, xmax, ymax])

        # Aplica transformações
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
        return len(self.images)
