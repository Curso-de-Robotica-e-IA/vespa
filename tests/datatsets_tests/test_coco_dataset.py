import pytest
from vespa.datasets.coco.coco_dataset import COCODataset
from vespa.datasets.coco.coco_transforms import get_coco_train_transforms, get_coco_test_transforms


@pytest.fixture
def coco_dataset(create_coco_dataset):
    """
    Cria uma instância do COCODataset usando a fixture create_coco_dataset.
    """
    root_dir, annotations_path = create_coco_dataset
    return COCODataset(
        root_dir=root_dir,
        ann_file=annotations_path,
        transforms=get_coco_train_transforms(),
    )


def test_coco_dataset_length(create_coco_annotations):
    """
    Verifica se o tamanho do dataset está correto.
    """
    dataset_path, annotations_path = create_coco_annotations
    dataset = COCODataset(root_dir=dataset_path, ann_file=annotations_path, transforms=None)
    assert len(dataset) == 5, "Tamanho do dataset COCO está incorreto."


def test_coco_dataset_image_dimensions(create_coco_annotations):
    """
    Verifica se as dimensões da imagem carregada estão corretas.
    """
    dataset_path, annotations_path = create_coco_annotations
    dataset = COCODataset(root_dir=dataset_path, ann_file=annotations_path, transforms=None)
    img, _ = dataset[0]
    assert img.shape == (100, 100, 3), f"Dimensão da imagem no COCODataset está incorreta: {img.shape}"


def test_coco_dataset_boxes_exist(create_coco_annotations):
    """
    Verifica se as bounding boxes existem no dataset.
    """
    dataset_path, annotations_path = create_coco_annotations
    dataset = COCODataset(root_dir=dataset_path, ann_file=annotations_path, transforms=None)
    _, target = dataset[0]
    assert len(target["boxes"]) > 0, "Bounding boxes no COCODataset não foram encontrados."
