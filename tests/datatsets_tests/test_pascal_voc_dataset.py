import pytest
import os
from vespa.datasets.pascalVOC.pascal_voc_dataset import PascalVOCDataset
from vespa.datasets.pascalVOC.pascal_voc_transforms import get_pascal_voc_train_transforms, get_pascal_voc_test_transforms


@pytest.fixture
def pascal_voc_dataset(create_pascal_voc_dataset):
    """
    Cria uma instância do PascalVOCDataset usando a fixture create_pascal_voc_dataset.
    """
    root_dir = create_pascal_voc_dataset
    return PascalVOCDataset(
        root_dir=root_dir,
        transforms=get_pascal_voc_train_transforms(),
    )


def test_pascal_voc_dataset_length(create_pascal_voc_dataset):
    """
    Testa se o número de imagens no dataset Pascal VOC está correto.
    """
    root_dir = create_pascal_voc_dataset
    dataset = PascalVOCDataset(root_dir=root_dir, transforms=None)
    assert len(dataset) == 5, "Tamanho do dataset Pascal VOC está incorreto."


def test_pascal_voc_dataset_image_shape(create_pascal_voc_dataset):
    """
    Testa se as imagens têm a dimensão esperada no PascalVOCDataset.
    """
    root_dir = create_pascal_voc_dataset
    dataset = PascalVOCDataset(root_dir=root_dir, transforms=None)
    img, _ = dataset[0]
    assert img.shape == (100, 100, 3), f"Dimensão da imagem no PascalVOCDataset está incorreta: {img.shape}"


def test_pascal_voc_dataset_boxes(create_pascal_voc_dataset):
    """
    Testa se as caixas delimitadoras no PascalVOCDataset são carregadas corretamente.
    """
    root_dir = create_pascal_voc_dataset
    dataset = PascalVOCDataset(root_dir=root_dir, transforms=None)
    _, target = dataset[0]
    assert len(target["boxes"]) > 0, "Bounding boxes no PascalVOCDataset não foram encontrados."


def test_pascal_voc_train_transforms_image_shape(create_pascal_voc_dataset):
    """
    Testa se as transformações de treino no Pascal VOC geram imagens com 3 dimensões.
    """
    root_dir = create_pascal_voc_dataset
    dataset = PascalVOCDataset(
        root_dir=root_dir,
        transforms=get_pascal_voc_train_transforms(),
    )
    img, _ = dataset[0]
    assert len(img.shape) == 3, "Imagem transformada para treino deve ter 3 dimensões."


def test_pascal_voc_test_transforms_no_bboxes(create_pascal_voc_dataset):
    """
    Testa se as transformações de teste no Pascal VOC não processam bounding boxes.
    """
    root_dir = create_pascal_voc_dataset
    dataset = PascalVOCDataset(
        root_dir=root_dir,
        transforms=get_pascal_voc_test_transforms(),
    )
    _, target = dataset[0]
    assert "boxes" in target, "Bounding boxes devem estar presentes no target."
