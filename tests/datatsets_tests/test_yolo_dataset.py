import os
import pytest
from vespa.datasets.yolo.yolo_dataset import YOLODataset
from vespa.datasets.yolo.yolo_transforms import get_yolo_train_transforms

@pytest.fixture
def yolo_dataset(create_yolo_dataset):
    """
    Cria uma instância do YOLODataset usando a fixture create_yolo_dataset.
    """
    root_dir = create_yolo_dataset
    return YOLODataset(
        root_dir=root_dir,
        txt_file="train.txt",
        image_size=100,
        transforms=get_yolo_train_transforms(),
    )


def test_yolo_dataset_length(create_dataset_path_train):
    dataset_path = create_dataset_path_train
    dataset = YOLODataset(root_dir=dataset_path, txt_file="train.txt", image_size=100, transforms=None)
    assert len(dataset) == 5, "Tamanho do dataset YOLO está incorreto."


def test_yolo_dataset_image_shape(create_dataset_path_train):
    dataset_path = create_dataset_path_train
    dataset = YOLODataset(root_dir=dataset_path, txt_file="train.txt", image_size=100, transforms=None)
    img, _ = dataset[0]
    assert img.shape == (100, 100, 3), f"Dimensão da imagem no YOLODataset está incorreta: {img.shape}"


def test_yolo_dataset_boxes(create_dataset_path_train):
    dataset_path = create_dataset_path_train
    dataset = YOLODataset(root_dir=dataset_path, txt_file="train.txt", image_size=100, transforms=None)
    _, target = dataset[0]
    assert len(target["boxes"]) > 0, "Bounding boxes no YOLODataset não foram encontrados."
