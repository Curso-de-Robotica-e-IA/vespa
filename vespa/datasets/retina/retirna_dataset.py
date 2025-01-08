import os
import torch
import cv2
import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm

class RetinaDataset(Dataset):
    def __init__(self, root_dir, txt_file, image_size, transforms=None, train_mode=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transforms = transforms
        self.train_mode = train_mode
        self.txt_file_path = os.path.join(root_dir, txt_file)

        # Lê o arquivo txt com as imagens
        with open(self.txt_file_path) as f:
            self.images = f.read().strip().split("\n")
        
        self.verify_images()
        
    
    def verify_images(self):
        before_size = self.__len__()
        confirms = [image for image in tqdm(self.images) if os.path.isfile(os.path.join(self.root_dir, image))]
        self.images = confirms

        if self.__len__() == 0:
            raise(Exception(f'No images found from paths in {self.txt_file_path}'))
        
        print(f'{self.__len__()} images read from {before_size}')


    def __getitem__(self, idx):
        # Carrega a imagem usando OpenCV (necessário para Albumentations)
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Carrega as anotações
        label_path = img_path.replace('/images', '/labels')
        label_path = label_path.replace('.jpg', '.txt')

        boxes = []
        labels = []

        with open(label_path) as f:
            for line in f.readlines():
                if line[0] != '\n':
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    labels.append(int(class_id))
                    
                    # YOLO usa coordenadas normalizadas, converte para formato [xmin, ymin, xmax, ymax]
                    xmin = (x_center - width / 2) * img.shape[1]
                    ymin = (y_center - height / 2) * img.shape[0]
                    xmax = (x_center + width / 2) * img.shape[1]
                    ymax = (y_center + height / 2) * img.shape[0]
                    boxes.append([xmin, ymin, xmax, ymax])
        
        # Converte caixas e rótulos para numpy
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Aplica transformações Albumentations, se houver
        if self.transforms and self.train_mode:
            augmented = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["labels"]
        elif self.transforms:
            img = self.transforms(img)
        
        # show_transforms(img, boxes)

        # Converte bboxes de volta para tensor se não estiver vazio
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # Se não houver caixas, cria um tensor vazio
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        if boxes.size(0) > 0:  # Verifica se há caixas antes de calcular a área
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.tensor([], dtype=torch.float32)  # Área vazia se não houver caixas
        
        # Converte para tensores PyTorch
        image_id = torch.tensor([idx])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
        }

        return img, target

    def __len__(self):
        return len(self.images)