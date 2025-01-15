import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET

import pytest
import torch
from PIL import Image

from vespa.methods.rcnn.model import RCNN


def create_image(images_dir, index):
    image_path = os.path.join(images_dir, f'image_{index}.jpg')
    image = Image.new(
        'RGB', (100, 100), color=(index * 20, index * 30, index * 40)
    )  # noqa
    image.save(image_path)


def create_label(labels_dir, index):
    label_path = os.path.join(labels_dir, f'image_{index}.txt')
    with open(label_path, 'w') as file:  # noqa
        file.write('0 0.1 0.1 0.1 0.1')


def create_pascal_voc_annotation(  # noqa
    annotations_dir, images_dir, index, class_name
):  # noqa
    """
    Cria uma anotação no formato Pascal VOC para uma imagem fictícia.

    Args:
        annotations_dir (str): Caminho para o diretório de anotações.
        images_dir (str): Caminho para o diretório de imagens.
        index (int): Índice da imagem/arquivo.
        class_name (str): Nome da classe para a anotação.
    """
    annotation_path = os.path.join(annotations_dir, f'image_{index}.xml')
    img_path = os.path.join(images_dir, f'image_{index}.jpg')

    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'images'

    filename = ET.SubElement(annotation, 'filename')
    filename.text = f'image_{index}.jpg'

    path = ET.SubElement(annotation, 'path')
    path.text = img_path

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Generated'

    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = '100'
    height = ET.SubElement(size, 'height')
    height.text = '100'
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    obj = ET.SubElement(annotation, 'object')
    name = ET.SubElement(obj, 'name')
    name.text = class_name  # Nome da classe

    pose = ET.SubElement(obj, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(obj, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(obj, 'difficult')
    difficult.text = '0'

    bndbox = ET.SubElement(obj, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = '20'
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '30'
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = '70'
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = '80'

    tree = ET.ElementTree(annotation)
    tree.write(annotation_path)


@pytest.fixture
def create_dataset_path_train(qtd_images=5):
    """
    Cria um diretório temporário com subdiretórios `images` e `labels`,
    além de um arquivo `train.txt`.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        images_dir = os.path.join(temp_dir, 'images')
        labels_dir = os.path.join(temp_dir, 'labels')
        os.makedirs(images_dir)
        os.makedirs(labels_dir)

        for i in range(qtd_images):
            create_image(images_dir, i)
            create_label(labels_dir, i)

        train_file_path = os.path.join(temp_dir, 'train.txt')
        with open(train_file_path, 'w') as train_file:  # noqa
            for i in range(qtd_images):
                train_file.write(f'images/image_{i}.jpg\n')

        return temp_dir
    except Exception as e:
        raise Exception(f'Erro ao criar arquivos temporários: {e}')


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
        'images': [
            {'id': i, 'file_name': f'images/image_{i}.jpg'} for i in range(5)
        ],  # noqa
        'annotations': [
            {
                'id': i,
                'image_id': i,
                'category_id': 1,
                'bbox': [10, 20, 30, 40],
            }
            for i in range(5)
        ],
        'categories': [{'id': 1, 'name': 'object'}],
    }
    annotations_path = os.path.join(root_dir, 'annotations.json')
    with open(annotations_path, 'w') as ann_file:  # noqa
        json.dump(annotations, ann_file)

    return root_dir, annotations_path


@pytest.fixture
def create_pascal_voc_dataset():
    """
    Cria um dataset temporário no formato Pascal VOC.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        annotations_dir = os.path.join(temp_dir, 'Annotations')
        images_dir = os.path.join(temp_dir, 'JPEGImages')
        os.makedirs(annotations_dir)
        os.makedirs(images_dir)

        # Criar 5 pares de imagens e anotações com classes variadas
        classes = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
        for i in range(5):
            create_image(images_dir, i)
            create_pascal_voc_annotation(
                annotations_dir,
                images_dir,
                i,
                class_name=classes[i % len(classes)],
            )

        return temp_dir
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e


@pytest.fixture
def tensor_image_fixture():
    rgb = torch.randn(1, 3, 600, 600)
    return rgb


@pytest.fixture
def rcnn_pretrained_fixture():
    return RCNN()


@pytest.fixture
def rcnn_sketch_fixture():
    return RCNN(pre_trained=False)
