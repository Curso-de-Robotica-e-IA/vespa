import numpy as np
import torch

class DatasetUtils:
    @staticmethod
    def yolo_to_retinanet(idx, image, label_path, transforms=None, train_mode=True):
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
        if transforms and train_mode:
            augmented = transforms(image=image, bboxes=boxes, labels=labels)
            img = augmented["image"]
            boxes = augmented["bboxes"]
            labels = augmented["labels"]
        elif transforms:
            img = transforms(image)
        
        # Converte bboxes de volta para tensor se não estiver vazio
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # Se não houver caixas, cria um tensor vazio
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

         # Verifica se há caixas antes de calcular a área
        if boxes.size(0) > 0: 
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            # Área vazia se não houver caixas
            area = torch.tensor([], dtype=torch.float32)  
        
        # Converte para tensores PyTorch
        image_id = torch.tensor([idx])

        return {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
        }