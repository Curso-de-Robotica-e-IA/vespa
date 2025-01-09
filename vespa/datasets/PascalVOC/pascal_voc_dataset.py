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
        self.class_to_idx = {}

        for file in os.listdir(os.path.join(root_dir, "Annotations")):
            if file.endswith(".xml"):
                self.annotation_paths.append(os.path.join(root_dir, "Annotations", file))
                self.image_paths.append(os.path.join(root_dir, "JPEGImages", file.replace(".xml", ".jpg")))

        self._generate_class_mapping()
                
    def _generate_class_mapping(self):
        """
        Gera o mapeamento dinâmico de classes para índices numéricos.
        """
        class_set = set()
        for annotation_path in self.annotation_paths:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                label = obj.find("name").text
                class_set.add(label)

        # Criar mapeamento ordenado para consistência
        self.class_to_idx = {label: idx for idx, label in enumerate(sorted(class_set))}

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
            if label not in self.class_to_idx:
                raise ValueError(f"Rótulo desconhecido: {label}")
            labels.append(self.class_to_idx[label])

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        if self.transforms:
            if "bboxes" in self.transforms.processors:  # Apenas para transformações que processam bboxes
                augmented = self.transforms(image=img, bboxes=boxes, labels=labels)
                img = augmented["image"]
                boxes = augmented["bboxes"]
                labels = augmented["labels"]
            else:
                augmented = self.transforms(image=img)  # Ignora caixas
                img = augmented["image"]

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
