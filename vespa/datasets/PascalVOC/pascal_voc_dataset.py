import os
import xml.etree.ElementTree as ET
import cv2
import torch
from vespa.datasets.base_dataset import BaseDataset


class PascalVOCDataset(BaseDataset):
    def __init__(self, root_dir, transforms=None):
        super().__init__(root_dir, transforms)
        self.image_paths = []
        self.annotation_paths = []

        for file in os.listdir(os.path.join(root_dir, 'Annotations')):
            if file.endswith('.xml'):
                self.annotation_paths.append(os.path.join(root_dir, 'Annotations', file))
                self.image_paths.append(os.path.join(root_dir, 'JPEGImages', file.replace('.xml', '.jpg')))

    def __getitem__(self, idx):
        annotation_path = self.annotation_paths[idx]
        img_path = self.image_paths[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(label)

            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

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
        return len(self.image_paths)
