import os

import cv2
import torch

from vespa.datasets.base_dataset import BaseDataset


class YOLODataset(BaseDataset):
    def __init__(self, root_dir, txt_file, image_size, transforms=None):
        """
        Inicializa o dataset YOLO.

        Args:
            root_dir (str): Diretório raiz das imagens e rótulos.
            txt_file (str): Nome do arquivo contendo a lista de imagens.
            image_size (int): Tamanho para redimensionar as imagens.
            transforms (callable, optional): Transformações a serem
                                    aplicadas nas imagens e anotações.
        """
        super().__init__(root_dir, transforms)
        self.image_size = image_size

        with open(os.path.join(root_dir, txt_file)) as f:  # noqa
            self.images = f.read().strip().split('\n')

    def __getitem__(self, idx):
        """
        Retorna uma amostra do dataset no formato esperado pelo PyTorch.

        Args:
            idx (int): Índice do item.

        Returns:
            tuple: Imagem transformada e dicionário com alvos (caixas e rótulos).
        """
        img_path = os.path.normpath(os.path.join(self.root_dir, self.images[idx]))
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Construa o caminho do rótulo
        label_path = img_path.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).replace('.jpg', '.txt')
        label_path = os.path.normpath(label_path)

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Rótulo não encontrado: {label_path}")

        boxes = []
        labels = []

        with open(label_path) as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(
                    float, line.strip().split()
                )
                labels.append(int(class_id))
                boxes.append([x_center, y_center, width, height])

        # Aplica transformações
        if self.transforms:
            if "bboxes" in self.transforms.processors:
                augmented = self.transforms(
                    image=img, bboxes=boxes, labels=labels
                )
                img = augmented["image"]
                boxes = augmented["bboxes"]
                labels = augmented["labels"]
            else:
                augmented = self.transforms(image=img)
                img = augmented["image"]

        # Converte boxes do formato YOLO para PASCAL VOC
        boxes = [
            [
                (box[0] - box[2] / 2) * img.shape[1],  # x_min
                (box[1] - box[3] / 2) * img.shape[0],  # y_min
                (box[0] + box[2] / 2) * img.shape[1],  # x_max
                (box[1] + box[3] / 2) * img.shape[0],  # y_max
            ]
            for box in boxes
        ]

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
        return len(self.images)
