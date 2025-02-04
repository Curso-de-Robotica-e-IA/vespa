import torch


class DatasetUtils:
    @staticmethod
    def yolo_to_retinanet(
        idx: int, image: torch.Tensor, boxes: list, labels: list
    ):
        converted_boxes = []
        for box in boxes:
            x_center, y_center, width, height = box
            xmin = int((x_center - width // 2) * image.shape[1])
            ymin = int((y_center - height // 2) * image.shape[2])
            xmax = int((x_center + width // 2) * image.shape[1])
            ymax = int((y_center + height // 2) * image.shape[2])
            converted_boxes.append([xmin, ymin, xmax, ymax])

        if len(converted_boxes) > 0:
            converted_boxes = torch.tensor(
                converted_boxes, dtype=torch.float32
            )
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            converted_boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        if converted_boxes.size(0) > 0:
            area = (converted_boxes[:, 2] - converted_boxes[:, 0]) * (
                converted_boxes[:, 3] - converted_boxes[:, 1]
            )
        else:
            area = torch.tensor([], dtype=torch.float32)

        target = {
            'boxes': converted_boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
        }

        return image, target

    @staticmethod
    def yolo_to_rcnn(): ...
