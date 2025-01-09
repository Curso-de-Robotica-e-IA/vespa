import os
import cv2
import torch
from pycocotools.coco import COCO
from vespa.datasets.base_dataset import BaseDataset


class COCODataset(BaseDataset):
    def __init__(self, root_dir, ann_file, transforms=None):
        """
        Inicializa o dataset COCO.

        Args:
            root_dir (str): Diretório raiz das imagens.
            ann_file (str): Caminho para o arquivo de anotações COCO.
            transforms (callable, optional): Transformações a serem aplicadas nas imagens e anotações.
        """
        super().__init__(root_dir, transforms)
        self.coco = COCO(ann_file)
        self.image_ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        """
        Retorna uma amostra do dataset no formato esperado pelo PyTorch.

        Args:
            idx (int): Índice do item.

        Returns:
            tuple: Imagem transformada e dicionário com alvos (caixas e rótulos).
        """
        image_id = self.image_ids[idx]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.root_dir, img_info["file_name"])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        for ann in annotations:
            xmin, ymin, width, height = ann["bbox"]
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann["category_id"])

        # Verifica se as transformações suportam `bboxes`
        if self.transforms:
            if self.transforms.processors.get("bboxes"):  # Verifica se bbox_params está definido
                augmented = self.transforms(image=img, bboxes=boxes, labels=labels)
                img = augmented["image"]
                boxes = augmented["bboxes"]
                labels = augmented["labels"]
            else:
                # Transformações de teste/validação sem `bboxes`
                augmented = self.transforms(image=img)
                img = augmented["image"]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return img, target


    def __len__(self):
        """
        Retorna o tamanho do dataset.

        Returns:
            int: Número de imagens no dataset.
        """
        return len(self.image_ids)
