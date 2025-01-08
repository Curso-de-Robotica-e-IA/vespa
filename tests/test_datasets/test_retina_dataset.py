import pytest
import shutil
import os
from PIL import Image
import tempfile

from vespa.datasets.retina.retirna_dataset import RetinaDataset
from vespa.datasets.retina.utils import get_train_transforms

def create_image(images_dir, index):
    image_path = os.path.join(images_dir, f"image_{index}.jpg")
    image = Image.new("RGB", (100, 100), color=(index * 20, index * 30, index * 40))
    image.save(image_path)

def create_label(labels_dir, index):
    label_path = os.path.join(labels_dir, f"image_{index}.txt")
    with open(label_path, 'w') as file:
        file.write('0 0.1 0.1 0.1 0.1')
    

@pytest.fixture
def create_dataset_path_train(qtd_images=5):
    # Cria um diretório temporário
    temp_dir = tempfile.mkdtemp()

    try:
        # Subdiretório para armazenar imagens
        images_dir = os.path.join(temp_dir, "images")
        labels_dir = os.path.join(temp_dir, "labels")
        os.makedirs(images_dir)
        os.makedirs(labels_dir)

        # Cria algumas imagens fictícias no subdiretório
        for i in range(qtd_images): 
            create_image(images_dir, i)
            create_label(labels_dir, i)

        # Cria um arquivo `train.txt` com a lista de caminhos das imagens
        train_file_path = os.path.join(temp_dir, "train.txt")
        with open(train_file_path, "w") as train_file:
            for i in range(qtd_images):
                train_file.write(f"images/image_{i}.jpg\n")

        # Retorna o caminho do dataset temporário
        return temp_dir
    except:
        raise Exception('Cant create temporary archives')

def destroy_dataset_path_train(path):
    shutil.rmtree(path)

def test_retina_instance(create_dataset_path_train):
    dataset_path = create_dataset_path_train
    data = RetinaDataset(dataset_path, 'train.txt', 1088, get_train_transforms('vespa/methods/retinanet/hyps.yaml'), train_mode=True)
    
    # Verifica se a primeira imagem é carregada corretamente
    first_sample = data[0]
    assert first_sample is not None

    destroy_dataset_path_train(dataset_path)