import numpy as np
import torch


class DatasetUtils:
    @staticmethod
    def yolo_to_retinanet(idx: int, image, label_path: str, transforms=None):
        boxes = []
        labels = []

        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line[0] != '\n':
                    class_id, x_center, y_center, width, height = map(
                        float, line.strip().split()
                    )
                    labels.append(int(class_id))

                    # YOLO usa coordenadas normalizadas, converte para formato
                    # [xmin, ymin, xmax, ymax]
                    xmin = int((x_center - width / 2) * image.shape[1])
                    ymin = int((y_center - height / 2) * image.shape[0])
                    xmax = int((x_center + width / 2) * image.shape[1])
                    ymax = int((y_center + height / 2) * image.shape[0])
                    boxes.append([xmin, ymin, xmax, ymax])

        # Converte caixas e rótulos para numpy
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Try apply Albumentations transforms
        try:
            augmented = transforms(image=image, bboxes=boxes, labels=labels)
            image = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['labels']
        except ValueError:
            # If val transform was used, don't need apply augmentations
            image = transforms(image=image)

        # Converte bboxes de volta para tensor se não estiver vazio
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # Se não houver caixas, cria um tensor vazio
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        if boxes.size(0) > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            # Área vazia se não houver caixas
            area = torch.tensor([], dtype=torch.float32)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
        }

        return image, target

    @staticmethod
    def yolo_to_rcnn(): ...
