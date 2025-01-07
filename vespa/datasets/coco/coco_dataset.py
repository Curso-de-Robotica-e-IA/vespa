import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, image_size, transforms=None, train_mode=True):
        """
        Dataset para o formato COCO.

        Args:
            root_dir (str): Diretório raiz contendo as imagens.
            annotation_file (str): Caminho para o arquivo de anotações JSON (COCO format).
            image_size (int): Tamanho para redimensionar as imagens (image_size x image_size).
            transforms (callable, optional): Transformações a serem aplicadas nas imagens e caixas.
            train_mode (bool): Indica se está em modo de treinamento (aplica transformações completas).
        """
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.image_size = image_size
        self.transforms = transforms
        self.train_mode = train_mode

        # Inicializa a API COCO
        self.coco = COCO(annotation_file)

        # Obtém os IDs das imagens
        self.image_ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        # Obtém o ID da imagem
        image_id = self.image_ids[idx]

        # Carrega os metadados da imagem
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))

        # Obtém as anotações da imagem
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            # Ignora anotações inválidas
            if 'bbox' in ann and 'category_id' in ann:
                x_min, y_min, width, height = ann['bbox']
                x_max = x_min + width
                y_max = y_min + height

                # Redimensiona as caixas para corresponder ao tamanho da imagem
                x_min = x_min * (self.image_size / img_info['width'])
                y_min = y_min * (self.image_size / img_info['height'])
                x_max = x_max * (self.image_size / img_info['width'])
                y_max = y_max * (self.image_size / img_info['height'])

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann['category_id'])

        # Converte para numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Aplica transformações, se houver
        if self.transforms and self.train_mode:
            augmented = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["labels"]
        elif self.transforms:
            img = self.transforms(img)

        # Converte caixas e rótulos para tensores
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # Se não houver caixas, cria tensores vazios
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        # Calcula as áreas das caixas
        if boxes.size(0) > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.tensor([], dtype=torch.float32)

        # Cria o dicionário de target
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),  # Para COCO, normalmente é 0 (não-crowd)
        }

        return img, target

    def __len__(self):
        return len(self.image_ids)