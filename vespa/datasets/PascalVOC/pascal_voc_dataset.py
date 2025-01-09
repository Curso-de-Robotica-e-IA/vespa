import os
import xml.etree.ElementTree as ET
import cv2
import torch
from vespa.datasets.base_dataset import BaseDataset


class PascalVOCDataset(BaseDataset):
    def __init__(self, root_dir, transforms=None):
        """
        Inicializa o dataset Pascal VOC.

        Args:
            root_dir (str): Diretório raiz contendo as imagens e anotações.
            transforms (callable, optional): Transformações a serem aplicadas nas imagens e anotações.
        """
        super().__init__(root_dir, transforms)
        self.image_paths = []
        self.annotation_paths = []

        annotations_dir = os.path.join(root_dir, "Annotations")
        images_dir = os.path.join(root_dir, "JPEGImages")

        for file in os.listdir(annotations_dir):
            if file.endswith(".xml"):
                self.annotation_paths.append(os.path.join(annotations_dir, file))
                self.image_paths.append(os.path.join(images_dir, file.replace(".xml", ".jpg")))

    def __getitem__(self, idx):
        """
        Retorna uma amostra do dataset no formato esperado pelo PyTorch.

        Args:
            idx (int): Índice do item.

        Returns:
            tuple: Imagem transformada e dicionário com alvos (caixas e rótulos).
        """
        annotation_path = self.annotation_paths[idx]
        img_path = self.image_paths[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            labels.append(label)

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
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
        """
        Retorna o número de amostras no dataset.

        Returns:
            int: Número de imagens no dataset.
        """
        return len(self.image_paths)
