import pytest
from vespa.datasets.yolo.yolo_dataset import YOLODataset
from vespa.datasets.yolo.yolo_transforms import get_yolo_train_transforms, get_yolo_test_transforms


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
    """
    Testa se o tamanho do dataset YOLO está correto.
    """
    dataset_path = create_dataset_path_train
    dataset = YOLODataset(root_dir=dataset_path, txt_file="train.txt", image_size=100, transforms=None)
    assert len(dataset) == 5, "Tamanho do dataset YOLO está incorreto."


def test_yolo_dataset_image_shape(create_dataset_path_train):
    """
    Testa se as imagens carregadas pelo YOLODataset têm as dimensões corretas.
    """
    dataset_path = create_dataset_path_train
    dataset = YOLODataset(root_dir=dataset_path, txt_file="train.txt", image_size=100, transforms=None)
    img, _ = dataset[0]
    assert img.shape == (100, 100, 3), f"Dimensão da imagem no YOLODataset está incorreta: {img.shape}"


def test_yolo_dataset_boxes(create_dataset_path_train):
    """
    Testa se as bounding boxes estão sendo carregadas corretamente.
    """
    dataset_path = create_dataset_path_train
    dataset = YOLODataset(root_dir=dataset_path, txt_file="train.txt", image_size=100, transforms=None)
    _, target = dataset[0]
    assert len(target["boxes"]) > 0, "Bounding boxes no YOLODataset não foram encontrados."


def test_yolo_train_transforms_image_shape(create_dataset_path_train):
    """
    Testa se as transformações de treino geram imagens com 3 dimensões.
    """
    dataset_path = create_dataset_path_train
    dataset = YOLODataset(
        root_dir=dataset_path,
        txt_file="train.txt",
        image_size=100,
        transforms=get_yolo_train_transforms(),
    )
    img, _ = dataset[0]
    assert len(img.shape) == 3, "Imagem transformada para treino deve ter 3 dimensões."


def test_yolo_train_transforms_boxes(create_dataset_path_train):
    """
    Testa se as bounding boxes são transformadas corretamente no treino.
    """
    dataset_path = create_dataset_path_train
    dataset = YOLODataset(
        root_dir=dataset_path,
        txt_file="train.txt",
        image_size=100,
        transforms=get_yolo_train_transforms(),
    )
    _, target = dataset[0]
    for box in target["boxes"]:
        assert all(0.0 <= coord <= 100.0 for coord in box), "Bounding boxes estão fora do intervalo esperado."



def test_yolo_test_transforms_image_shape(create_dataset_path_train):
    """
    Testa se as transformações de teste geram imagens com 3 dimensões.
    """
    dataset_path = create_dataset_path_train
    dataset = YOLODataset(
        root_dir=dataset_path,
        txt_file="train.txt",
        image_size=100,
        transforms=get_yolo_test_transforms(),
    )
    img, _ = dataset[0]
    assert len(img.shape) == 3, "Imagem transformada para teste deve ter 3 dimensões."


def test_yolo_test_transforms_no_bboxes(create_dataset_path_train):
    """
    Testa se as transformações de teste não incluem bounding boxes.
    """
    dataset_path = create_dataset_path_train
    dataset = YOLODataset(
        root_dir=dataset_path,
        txt_file="train.txt",
        image_size=100,
        transforms=get_yolo_test_transforms(),
    )
    _, target = dataset[0]
    assert "boxes" in target, "Bounding boxes devem estar presentes no target, mesmo sem transformações."



def test_yolo_test_transforms_no_bboxes(create_dataset_path_train):
    """
    Testa se as transformações de teste não incluem bounding boxes.
    """
    dataset_path = create_dataset_path_train
    dataset = YOLODataset(
        root_dir=dataset_path,
        txt_file="train.txt",
        image_size=100,
        transforms=get_yolo_test_transforms(),
    )
    _, target = dataset[0]
    assert "boxes" in target, "Bounding boxes devem estar presentes no target, mesmo sem transformações."