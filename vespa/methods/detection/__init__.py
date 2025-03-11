"""
Detection Models Module

This module provides a unified interface for training and running inference on
object detection models, including YOLO (ONNX), RCNN, and RetinaNet.

Usage:
    - Train a model:
        from vespa.methods.detection import train_model
        train_model(
            "yolo",
            "path/to/dataset",
            "yolo",
            batch_size=8,
            epochs=25,
            device="cuda"
            )

    - Validate a model:
        from vespa.methods.detection import validate_model
        validate_model(
            "rcnn",
            "path/to/dataset",
            "yolo",
            batch_size=8,
            device="cuda"
            )

    - Test a model:
        from vespa.methods.detection import test_model
        test_model(
            "retinanet",
            "path/to/dataset",
            "yolo",
            batch_size=8,
            device="cuda"
            )

    - List available models:
        from vespa.methods.detection import available_models
        print(available_models())

    - Run inference:
        from vespa.methods.detection import run_inference
        predictions = run_inference("rcnn", "path/to/image.jpg")

Functions:
    - train_model(
        model_name: str,
        dataset_path: str,
        dataset_format: str,
        **dataset_kwargs
        ): Trains a specified detection model.
    - validate_model(
        model_name: str,
        dataset_path: str,
        dataset_format: str,
        **dataset_kwargs
        ): Validates a specified detection model.
    - test_model(
        model_name: str,
        dataset_path: str,
        dataset_format: str,
        **dataset_kwargs
        ): Tests a specified detection model.
    - available_models(): Returns a list of available detection models.
    - run_inference(
        model_name: str,
        image_path: str
        ): Runs inference using the specified detection model.

Models:
    - YOLO (ONNX-based)
    - RCNN
    - RetinaNet

Author:
    - Vespa AI Team
"""

import json
import os
import xml.etree.ElementTree as ET

import cv2
import yaml
from torchvision.transforms import ToTensor

from vespa.datasets.detection_formats import load_dataset
from vespa.datasets.detection_formats.coco.coco_transforms import (
    get_coco_test_transforms,
    get_coco_train_transforms,
)
from vespa.datasets.detection_formats.pascal_voc.pascal_voc_transforms import (
    get_pascal_voc_test_transforms,
    get_pascal_voc_train_transforms,
)
from vespa.datasets.detection_formats.yolo.yolo_transforms import (
    get_yolo_test_transforms,
    get_yolo_train_transforms,
)

from .rcnn.model import RCNN
from .retinanet.model import RetinaNet


def get_num_classes(dataset_path: str, dataset_format: str) -> int:
    """
    Retrieve the number of classes from the dataset configuration file.

    Args:
        dataset_path (str): Path to the dataset directory.
        dataset_format (str): Format of the dataset
        ('yolo', 'coco', 'pascal_voc').

    Returns:
        int: Number of classes in the dataset.

    Raises:
        ValueError: If the dataset format is unsupported.
    """
    if dataset_format == 'yolo':
        yaml_file = os.path.join(dataset_path, 'data.yaml')
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(
                f'YOLO dataset config file not found: {yaml_file}'
            )

        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        class_names = data.get('names', [])
        if not class_names:
            raise ValueError(
                'No class names found in YOLO dataset configuration.'
            )

        return len(class_names) + 1  # Include background class

    elif dataset_format == 'coco':
        json_file = os.path.join(dataset_path, 'annotations.json')
        if not os.path.exists(json_file, 'r', encoding='utf-8'):
            raise FileNotFoundError(
                f'COCO dataset config file not found: {json_file}'
            )

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        categories = data.get('categories', [])
        if not categories:
            raise ValueError('No categories found in COCO dataset.')

        return len(categories) + 1  # Include background class

    elif dataset_format == 'pascal_voc':
        xml_file = os.path.join(dataset_path, 'classes.xml')
        if not os.path.exists(xml_file):
            raise FileNotFoundError(
                f'Pascal VOC dataset config file not found: {xml_file}'
            )

        tree = ET.parse(xml_file)
        root = tree.getroot()

        class_elements = root.findall('class')
        if not class_elements:
            raise ValueError('No class elements found in Pascal VOC XML file.')

        return len(class_elements) + 1  # Include background class

    else:
        raise ValueError('Unsupported dataset format.')


def get_train_tranforms(dataset_format: str):
    """
    Get the appropriate transforms for the specified dataset format.
    """
    if dataset_format == 'yolo':
        return get_yolo_train_transforms()
    elif dataset_format == 'coco':
        return get_coco_train_transforms()
    elif dataset_format == 'pascal_voc':
        return get_pascal_voc_train_transforms()


def get_test_tranforms(dataset_format: str):
    """
    Get the appropriate transforms for the specified dataset format.
    """
    if dataset_format == 'yolo':
        return get_yolo_test_transforms()
    elif dataset_format == 'coco':
        return get_coco_test_transforms()
    elif dataset_format == 'pascal_voc':
        return get_pascal_voc_test_transforms()


def train_model(
    model_name: str, dataset_path: str, dataset_format: str, **dataset_kwargs
):
    """
    Train a specified detection model.

    Args:
        model_name (str): The name of the model to train
        ('yolo', 'rcnn', 'retinanet').
        dataset_path (str): Path to the dataset.
        dataset_format (str): Format of the dataset
        ('yolo', 'coco', 'pascal_voc').

    Raises:
        ValueError: If an unsupported model name is provided.
    """

    train_transforms = get_train_tranforms(dataset_format)

    train_dataset = load_dataset(
        dataset_path,
        dataset_format,
        model_name=model_name,
        transforms=train_transforms,
        **dataset_kwargs,
    )

    num_classes = get_num_classes(dataset_path, dataset_format)

    if num_classes is None:
        raise ValueError(
            f'Failed to determine the number of classes for dataset: {dataset_path}'  # noqa
        )

    print(f'Using {num_classes} classes for training {model_name}')

    if model_name == 'retinanet':
        model = RetinaNet(num_classes=num_classes)
    elif model_name == 'rcnn':
        model = RCNN(num_classes=num_classes)
    else:
        raise ValueError('Unsupported model name.')

    model.fit(
        train_dataset,
        batch_size=dataset_kwargs.get('batch_size', 4),
        epochs=dataset_kwargs.get('epochs', 20),
        device=dataset_kwargs.get('device', 'cuda'),
    )


def validate_model(
    model_name: str, dataset_path: str, dataset_format: str, **dataset_kwargs
):
    """
    Validate a trained model using a validation dataset.
    """
    test_transforms = get_test_tranforms(dataset_format)
    val_dataset = load_dataset(
        dataset_path,
        dataset_format,
        model_name=model_name,
        transforms=test_transforms,
        **dataset_kwargs,
    )
    if model_name == 'retinanet':
        model = RetinaNet()
    elif model_name == 'rcnn':
        model = RCNN()
    else:
        raise ValueError('Unsupported model name.')
    return model.valid(val_dataset, batch_size=4, device='cuda')


def available_models():
    """Returns a list of available detection models."""
    return ['onnx_yolo', 'rcnn', 'retinanet']


def run_inference(model_name: str, image_path: str):
    """
    Runs inference using the specified detection model.

    Args:
        model_name (str): The name of the model to use
        ('onnx_yolo', 'rcnn', 'retinanet').
        image_path (str): Path to the input image.

    Returns:
        dict: Inference results from the selected model.

    Raises:
        ValueError: If an unsupported model name is provided.
    """

    # Load image and convert to tensor
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'Error loading image: {image_path}')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = ToTensor()(image)  # Convert to Tensor
    image = image.unsqueeze(0)  # Add batch dimension

    # Load the correct model
    if model_name == 'retinanet':
        model = RetinaNet()
    elif model_name == 'rcnn':
        model = RCNN()
    else:
        raise ValueError('Unsupported model name.')

    # Perform inference
    return model.predict(image, device='cuda')


__all__ = [
    'RetinaNet',
    'train_model',
    'validate_model',
    'available_models',
    'run_inference',
]
