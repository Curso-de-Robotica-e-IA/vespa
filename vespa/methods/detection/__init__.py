from .onnx_yolo.model import YOLO
from .rcnn.model import RCNN
from .retinanet.model import RetinaNet

def train_model(model_name: str, dataset_path: str, dataset_format: str):
    if model_name == "yolo":
        model = YOLO()
    elif model_name == "rcnn":
        model = RCNN()
    elif model_name == "retinanet":
        model = RetinaNet()
    else:
        raise ValueError("Unsupported model name.")

    return model.fit(dataset_path, dataset_format)

def available_models():
    return ["yolo", "rcnn", "retinanet"]

def run_inference(model_name: str, image_path: str):
    if model_name == "yolo":
        model = YOLO()
    elif model_name == "rcnn":
        model = RCNN()
    elif model_name == "retinanet":
        model = RetinaNet()
    else:
        raise ValueError("Unsupported model name.")

    return model.predict(image_path)

__all__ = ["YOLO", "RCNN", "RetinaNet", "train_model", "available_models", "run_inference"]