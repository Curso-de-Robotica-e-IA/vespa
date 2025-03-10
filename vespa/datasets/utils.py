import torch
from torch import tensor, empty, zeros, int64, float32

def yolo_preprocess(image: torch.Tensor, boxes: list, labels: list):
    """
    Preprocess YOLO format bounding boxes into a format suitable for conversion.
    """
    converted_boxes = []
    img_h, img_w = image.shape[1], image.shape[2]  # Altura e largura

    for box in boxes:
        x_center, y_center, width, height = box

        xmin = int((x_center - width / 2) * img_w)
        ymin = int((y_center - height / 2) * img_h)
        xmax = int((x_center + width / 2) * img_w)
        ymax = int((y_center + height / 2) * img_h)

        if xmin >= xmax or ymin >= ymax:
            print(f'⚠️ Bounding box inválida removida: {[xmin, ymin, xmax, ymax]}')
            continue

        converted_boxes.append([xmin, ymin, xmax, ymax])

    if len(converted_boxes) > 0:
        converted_boxes = tensor(converted_boxes, dtype=float32)
        labels = tensor(labels, dtype=int64)
        area = (converted_boxes[:, 2] - converted_boxes[:, 0]) * (converted_boxes[:, 3] - converted_boxes[:, 1])
    else:
        converted_boxes = empty((0, 4), dtype=float32)
        labels = empty((0,), dtype=int64)
        area = tensor([], dtype=float32)
    
    return converted_boxes, labels, area

def yolo_to_retinanet(idx: int, image: torch.Tensor, boxes: list, labels: list):
    """
    Converts YOLO format bounding boxes to RetinaNet format.
    """
    converted_boxes, labels, area = yolo_preprocess(image, boxes, labels)
    target = {
        'boxes': converted_boxes,
        'labels': labels,
        'image_id': tensor([idx]),
        'area': area,
    }
    return image, target

def yolo_to_rcnn(idx: int, image: torch.Tensor, boxes: list, labels: list):
    """
    Converts YOLO format bounding boxes to RCNN format.
    """
    converted_boxes, labels, area = yolo_preprocess(image, boxes, labels)
    target = {
        'boxes': converted_boxes,
        'labels': labels,
        'image_id': tensor([idx]),
        'area': area,
        'iscrowd': zeros((len(converted_boxes),), dtype=int64),
    }
    return image, target