"""
Detection Models Module

This module provides a unified interface for training and running inference on
object detection models, including YOLO (ONNX), RCNN, and RetinaNet.

Usage:
    - Train a model:
        from vespa.methods.detection import train_model
        train_model("yolo", "path/to/dataset", "yolo")

    - List available models:
        from vespa.methods.detection import available_models
        print(available_models())

    - Run inference:
        from vespa.methods.detection import run_inference
        predictions = run_inference("rcnn", "path/to/image.jpg")

Functions:
    - train_model(model_name: str, dataset_path: str, dataset_format: str): Trains a specified detection model.
    - available_models(): Returns a list of available detection models.
    - run_inference(model_name: str, image_path: str): Runs inference using the specified detection model.

Models:
    - YOLO (ONNX-based)
    - RCNN
    - RetinaNet

Author:
    - Vespa AI Team
"""

from .onnx_yolo.model import ONNX_YOLO
from .rcnn.model import RCNN
from .retinanet.model import RetinaNet

def train_model(model_name: str, dataset_path: str, dataset_format: str):
    """
    Trains a specified detection model.

    Args:
        model_name (str): The name of the model to train ('yolo', 'rcnn', 'retinanet').
        dataset_path (str): Path to the dataset for training.
        dataset_format (str): Dataset format (e.g., 'yolo', 'coco', 'pascal_voc').

    Returns:
        Training results from the selected model.

    Raises:
        ValueError: If an unsupported model name is provided.
    """
    if model_name == "onnx_yolo":
        model = ONNX_YOLO()
    elif model_name == "rcnn":
        model = RCNN()
    elif model_name == "retinanet":
        model = RetinaNet()
    else:
        raise ValueError("Unsupported model name.")

    return model.fit(dataset_path, dataset_format)

def available_models():
    """
    Returns a list of available detection models.

    Returns:
        list: ["onnx_yolo", "rcnn", "retinanet"]
    """
    return ["onnx_yolo", "rcnn", "retinanet"]

def run_inference(model_name: str, image_path: str):
    """
    Runs inference using the specified detection model.

    Args:
        model_name (str): The name of the model to use ('onnx_yolo', 'rcnn', 'retinanet').
        image_path (str): Path to the input image.

    Returns:
        dict: Inference results from the selected model.

    Raises:
        ValueError: If an unsupported model name is provided.
    """
    if model_name == "onnx_yolo":
        model = ONNX_YOLO()
    elif model_name == "rcnn":
        model = RCNN()
    elif model_name == "retinanet":
        model = RetinaNet()
    else:
        raise ValueError("Unsupported model name.")

    return model.predict(image_path)

__all__ = ["ONNX_YOLO", "RCNN", "RetinaNet", "train_model", "available_models", "run_inference"]