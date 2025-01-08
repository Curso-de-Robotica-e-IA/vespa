import os
import pytest
import tempfile
import shutil
import json
import xml.etree.ElementTree as ET
from PIL import Image


def create_image(images_dir, index):
    image_path = os.path.join(images_dir, f"image_{index}.jpg")
    image = Image.new("RGB", (100, 100), color=(index * 20, index * 30, index * 40))
    image.save(image_path)


def create_label(labels_dir, index):
    label_path = os.path.join(labels_dir, f"image_{index}.txt")
    with open(label_path, "w") as file:
        file.write("0 0.1 0.1 0.1 0.1")


@pytest.fixture
def create_dataset_path_train(qtd_images=5):
    """
    Cria um diretório temporário com subdiretórios `images` e `labels`,
    além de um arquivo `train.txt`.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        images_dir = os.path.join(temp_dir, "images")
        labels_dir = os.path.join(temp_dir, "labels")
        os.makedirs(images_dir)
        os.makedirs(labels_dir)

        for i in range(qtd_images):
            create_image(images_dir, i)
            create_label(labels_dir, i)

        train_file_path = os.path.join(temp_dir, "train.txt")
        with open(train_file_path, "w") as train_file:
            for i in range(qtd_images):
                train_file.write(f"images/image_{i}.jpg\n")

        return temp_dir
    except Exception as e:
        raise Exception(f"Erro ao criar arquivos temporários: {e}")


def destroy_dataset_path_train(path):
    """
    Remove o diretório temporário.
    """
    shutil.rmtree(path)


@pytest.fixture
def create_coco_annotations(create_dataset_path_train):
    """
    Extende a fixture base para adicionar um arquivo JSON no formato COCO.
    """
    root_dir = create_dataset_path_train
    annotations = {
        "images": [{"id": i, "file_name": f"images/image_{i}.jpg"} for i in range(5)],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [10, 20, 30, 40],
            }
            for i in range(5)
        ],
        "categories": [{"id": 1, "name": "object"}],
    }
    annotations_path = os.path.join(root_dir, "annotations.json")
    with open(annotations_path, "w") as ann_file:
        json.dump(annotations, ann_file)

    return root_dir, annotations_path


@pytest.fixture
def create_pascal_voc_annotations(create_dataset_path_train):
    """
    Extende a fixture base para adicionar arquivos XML no formato Pascal VOC.
    """
    root_dir = create_dataset_path_train
    annotations_dir = os.path.join(root_dir, "Annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    for i in range(5):
        annotation = ET.Element("annotation")
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "object"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = "10"
        ET.SubElement(bndbox, "ymin").text = "20"
        ET.SubElement(bndbox, "xmax").text = "40"
        ET.SubElement(bndbox, "ymax").text = "60"

        tree = ET.ElementTree(annotation)
        tree.write(os.path.join(annotations_dir, f"image_{i}.xml"))

    return root_dir


@pytest.fixture
def create_kitti_annotations(create_dataset_path_train):
    """
    Extende a fixture base para adicionar arquivos de rótulos no formato KITTI.
    """
    root_dir = create_dataset_path_train
    labels_dir = os.path.join(root_dir, "labels")

    for i in range(5):
        label_path = os.path.join(labels_dir, f"image_{i}.txt")
        with open(label_path, "w") as label_file:
            label_file.write("Car 0 0 0 10 20 30 40 0 0 0 0 0 0 0\n")

    return root_dir, labels_dir
