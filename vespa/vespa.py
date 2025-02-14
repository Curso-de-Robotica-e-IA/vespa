"""
Vespa AI Model Manager

This module provides functions to manage AI model tasks, including setting task types, 
selecting dataset formats, training models, and running inference.

Usage:
    - Set a task type:
        set_task("detection")

    - Set dataset format:
        set_dataset_format("yolo")

    - Train a model:
        train("path/to/dataset")

    - Run inference:
        predict("path/to/image.jpg")

Functions:
    - set_task(task_type: str): Sets the type of AI task.
    - set_dataset_format(dataset_format: str): Sets the dataset format.
    - set_model(model_name: str): Specifies the model to use.
    - train(dataset_path: str): Trains a model based on the selected task.
    - predict(image_path: str): Runs inference using the selected model.
    - list_models(): Returns available models for detection.
    - list_dataset_formats(): Returns supported dataset formats.
    - train_model(model_name: str, dataset_path: str, dataset_format: str): Trains a specific model.
    - run_inference(model_name: str, image_path: str): Runs inference using a specified model.

Dependencies:
    - Requires `vespa.methods` for model training and inference.
    - Requires `vespa.datasets` for dataset format handling.

Author:
    - Vespa AI Team
"""

from vespa.methods import detection, classification, regression
from vespa.datasets import detection_formats

# Global variables to store selected task, model, and dataset format
_task = None
_model = None
_dataset_format = None

def set_task(task_type: str):
    """
    Sets the type of AI task (classification, regression, or detection).

    Args:
        task_type (str): The type of task. Must be one of ['classification', 'regression', 'detection'].

    Raises:
        ValueError: If an invalid task type is provided.
    """
    global _task
    if task_type not in ["classification", "regression", "detection"]:
        raise ValueError("Invalid task type. Choose from 'classification', 'regression', 'detection'.")
    _task = task_type

def set_dataset_format(dataset_format: str):
    """
    Sets the dataset format (YOLO, COCO, or Pascal VOC).

    Args:
        dataset_format (str): The dataset format.

    Raises:
        ValueError: If an unsupported dataset format is provided.
    """
    global _dataset_format
    if dataset_format not in ["yolo", "coco", "pascal_voc"]:
        raise ValueError("Invalid dataset format.")
    _dataset_format = dataset_format

def set_model(model_name: str):
    """
    Specifies the model to use for training or inference.

    Args:
        model_name (str): The model name.
    """
    global _model
    _model = model_name

def train(dataset_path: str):
    """
    Trains a model using the selected task and dataset format.

    Args:
        dataset_path (str): Path to the dataset.

    Returns:
        str: Training results.

    Raises:
        ValueError: If no task type is set.
    """
    if _task == "detection":
        dataset_kwargs = {
            "txt_file": "train.txt",
        }
        return detection.train_model(_model, dataset_path, _dataset_format, **dataset_kwargs)
    else:
        raise ValueError("No task type set. Use set_task().")
    
def validate(dataset_path: str):
    """
    Validates a model using the selected task and dataset format.

    Args:
        dataset_path (str): Path to the dataset.

    Returns:
        str: Validation results.

    Raises:
        ValueError: If no task type is set.
    """
    if _task == "detection":
        dataset_kwargs = {
            "txt_file": "val.txt",
        }
        return detection.validate_model(_model, dataset_path, _dataset_format, **dataset_kwargs)
    else:
        raise ValueError("No task type set. Use set_task().")

def predict(image_path: str):
    """
    Runs inference using the selected model.

    Args:
        image_path (str): Path to the image for inference.

    Returns:
        dict: Inference results.

    Raises:
        ValueError: If no task type is set.
    """
    if _task == "detection":
        return detection.run_inference(_model, image_path)
    else:
        raise ValueError("No task type set. Use set_task().")

def list_models():
    """
    Returns available models for the selected task.

    Returns:
        dict: Dictionary of available models.
    """
    return {
        "detection": detection.available_models(),
    }

def list_dataset_formats():
    """
    Returns supported dataset formats.

    Returns:
        list: List of supported dataset formats.
    """
    return ["yolo", "coco", "pascal_voc"]