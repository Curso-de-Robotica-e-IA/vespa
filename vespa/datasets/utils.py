from torch import Tensor, empty, float32, int64, tensor


def yolo_to_retinanet(idx: int, image: Tensor, boxes: list, labels: list):
    """
    Converts YOLO format bounding boxes to RetinaNet format.
    Args:
        idx (int): The index of the image.
        image (Tensor): The image tensor with shape (C, H, W).
        boxes (list): A list of bounding boxes in YOLO format,
                      where each box is represented as
                      [x_center, y_center, width, height].
        labels (list): A list of labels corresponding to the bounding boxes.
    Returns:
        Tuple[Tensor, dict]: A tuple containing the image tensor and a
        dictionary with the following keys:\n
            - 'boxes' (Tensor): Converted bounding boxes in RetinaNet
            format [xmin, ymin, xmax, ymax].\n
            - 'labels' (Tensor): Tensor of labels.\n
            - 'image_id' (Tensor): Tensor containing the image index.\n
            - 'area' (Tensor): Tensor containing the area of each bounding box.
    """
    converted_boxes = []
    img_h, img_w = image.shape[1], image.shape[2]  # Altura e largura corretas

    for box in boxes:
        x_center, y_center, width, height = box

        xmin = int((x_center - width / 2) * img_w)
        ymin = int((y_center - height / 2) * img_h)
        xmax = int((x_center + width / 2) * img_w)
        ymax = int((y_center + height / 2) * img_h)

        # Garante que as coordenadas estejam corretas
        if xmin >= xmax or ymin >= ymax:
            print(
                f'⚠️ Bounding box inválida removida: {[xmin, ymin, xmax, ymax]}'
            )
            continue  # Ignora caixas inválidas

        converted_boxes.append([xmin, ymin, xmax, ymax])

    if len(converted_boxes) > 0:
        converted_boxes = tensor(converted_boxes, dtype=float32)
        labels = tensor(labels, dtype=int64)
        area = converted_boxes[:, 2] - converted_boxes[:, 0]
        area *= converted_boxes[:, 3] - converted_boxes[:, 1]
    else:
        converted_boxes = empty((0, 4), dtype=float32)
        labels = empty((0,), dtype=int64)
        area = tensor([], dtype=float32)

    target = {
        'boxes': converted_boxes,
        'labels': labels,
        'image_id': tensor([idx]),
        'area': area,
    }

    return image, target
