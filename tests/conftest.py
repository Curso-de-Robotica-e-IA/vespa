import pytest
import torch

from vespa.datasets.detection_formats.yolo.yolo_dataset import YOLODataset
from vespa.datasets.detection_formats.yolo.yolo_transforms import (
    get_yolo_test_transforms,
    get_yolo_train_transforms,
)


@pytest.fixture
def root_path_dataset_yolo_format():
    return './assets/datasets/detection/yolo'


@pytest.fixture
def yolo_dataset_train_retina(root_path_dataset_yolo_format):
    """
    Cria uma instância do YOLODataset usando a fixture create_yolo_dataset.
    """
    return YOLODataset(
        root_dir=root_path_dataset_yolo_format,
        txt_file='train.txt',
        image_size=416,
        transforms=get_yolo_train_transforms(),
        model_name='retinanet',
    )


@pytest.fixture
def yolo_dataset_test_retina(root_path_dataset_yolo_format):
    """
    Cria uma instância do YOLODataset usando a fixture create_yolo_dataset.
    """
    return YOLODataset(
        root_dir=root_path_dataset_yolo_format,
        txt_file='val.txt',
        image_size=416,
        transforms=get_yolo_test_transforms(),
        model_name='retinanet',
    )


@pytest.fixture
def yolo_dataset_train_rcnn(root_path_dataset_yolo_format):
    """
    Cria uma instância do YOLODataset usando a fixture create_yolo_dataset.
    """
    return YOLODataset(
        root_dir=root_path_dataset_yolo_format,
        txt_file='train.txt',
        image_size=416,
        transforms=get_yolo_train_transforms(),
        model_name='rcnn',
    )


@pytest.fixture
def yolo_dataset_test_rcnn(root_path_dataset_yolo_format):
    """
    Cria uma instância do YOLODataset usando a fixture create_yolo_dataset.
    """
    return YOLODataset(
        root_dir=root_path_dataset_yolo_format,
        txt_file='val.txt',
        image_size=416,
        transforms=get_yolo_test_transforms(),
        model_name='rcnn',
    )


@pytest.fixture
def tensor_image_fixture():
    rgb = torch.randn(1, 3, 600, 600)
    return rgb
