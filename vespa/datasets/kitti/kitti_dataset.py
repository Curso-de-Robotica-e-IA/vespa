import os
import torch
import cv2
from vespa.datasets.base_dataset import BaseDataset


class KITTIDataset(BaseDataset):
    def __init__(self, root_dir, label_dir, transforms=None):
        """
        Inicializa o dataset KITTI.

        Args:
            root_dir (str): Diretório raiz das imagens.
            label_dir (str): Diretório contendo os arquivos de rótulos.
            transforms (callable, optional): Transformações a serem aplicadas nas imagens e anotações.
        """
        super().__init__(root_dir, transforms)
        self.label_dir = label_dir
        self.image_paths = sorted(
            [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
        )
        self.label_paths = sorted(
            [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.txt')]
        )

    def __getitem__(self, idx):
        """
        Retorna uma amostra do dataset no formato esperado pelo PyTorch.

        Args:
            idx (int): Índice do item.

        Returns:
            tuple: Imagem transformada e dicionário com alvos (caixas e rótulos).
        """
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Rótulo não encontrado para {img_path}")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        with open(label_path, 'r') as f:
            for line in f.readlines():
                values = line.strip().split()
                class_name = values[0]  # Nome da classe
                xmin = float(values[4])
                ymin = float(values[5])
                xmax = float(values[6])
                ymax = float(values[7])

                labels.append(class_name)
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
