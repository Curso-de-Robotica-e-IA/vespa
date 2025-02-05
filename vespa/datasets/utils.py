from torch import Tensor, tensor, empty, float32, int64


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
    for box in boxes:
        x_center, y_center, width, height = box
        xmin = int((x_center - width // 2) * image.shape[1])
        ymin = int((y_center - height // 2) * image.shape[2])
        xmax = int((x_center + width // 2) * image.shape[1])
        ymax = int((y_center + height // 2) * image.shape[2])
        converted_boxes.append([xmin, ymin, xmax, ymax])

    if len(converted_boxes) > 0:
        converted_boxes = tensor(
            converted_boxes, dtype=float32
        )
        labels = tensor(labels, dtype=int64)
    else:
        converted_boxes = empty((0, 4), dtype=float32)
        labels = empty((0,), dtype=int64)

    if converted_boxes.size(0) > 0:
        area = (converted_boxes[:, 2] - converted_boxes[:, 0]) * (
            converted_boxes[:, 3] - converted_boxes[:, 1]
        )
    else:
        area = tensor([], dtype=float32)

    target = {
        'boxes': converted_boxes,
        'labels': labels,
        'image_id': tensor([idx]),
        'area': area,
    }

    return image, target