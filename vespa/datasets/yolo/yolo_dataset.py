import os

import cv2
from tqdm import tqdm

from vespa.datasets.base_dataset import BaseDataset


class YOLODataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        txt_file: str,
        image_size: int,
        transforms=None,
        model: str = None,
    ):
        """
        Inicializa o dataset YOLO.

        Args:
            root_dir (str): Diretório raiz das imagens e rótulos.
            txt_file (str): Nome do arquivo contendo a lista de imagens.
            image_size (int): Tamanho para redimensionar as imagens.
            transforms (callable, optional): Transformações a serem
                                    aplicadas nas imagens e anotações.
        """
        super().__init__(root_dir, transforms, model)
        self.image_size = image_size
        self.txt_file_path = os.path.join(root_dir, txt_file)

        # Lê o arquivo txt com as imagens e labels
        with open(self.txt_file_path, 'r', encoding='utf-8') as f:
            self.images = f.read().strip().split('\n')

        self.verify_images()

    def verify_images(self):
        before_size = len(self.images)
        confirms = [
            os.path.normpath(os.path.join(self.root_dir, image))
            for image in tqdm(self.images)
            if os.path.isfile(
                os.path.normpath(os.path.join(self.root_dir, image))
            )
            and os.path.isfile(
                os.path.normpath(
                    os.path.join(
                        self.root_dir,
                        image.replace('/images/', '/labels/').replace(
                            '.jpg', '.txt'
                        ),
                    )
                )
            )
        ]

        self.images = confirms
        current_size = len(self.images)
        if current_size == 0:
            raise (Exception(f'No images found: {self.txt_file_path}'))

        print(f'{current_size} images read from {before_size}')

    def __getitem__(self, idx):
        """
        Retorna uma amostra do dataset no formato esperado pelo PyTorch.

        Args:
            idx (int): Índice do item.

        Returns:
            tuple: Imagem transformada e dicionário com alvos
                   (caixas e rótulos).
        """
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f'Image not found: {img_path}')

        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Construa o caminho do rótulo
        label_path = img_path.replace(
            os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
        ).replace('.jpg', '.txt')  # noqa
        label_path = os.path.normpath(label_path)

        if not os.path.exists(label_path):
            raise FileNotFoundError(f'Label not found: {label_path}')

        boxes = []
        labels = []

        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line[0] != '\n':
                    class_id, x_center, y_center, width, height = map(
                        float, line.strip().split()
                    )
                    labels.append(int(class_id))
                    boxes.append([x_center, y_center, width, height])

        # Try apply Albumentations transforms
        try:
            augmented = self.transforms(
                image=image, bboxes=boxes, labels=labels
            )
            image = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['labels']
        except ValueError:
            # If val transform was used, don't need apply augmentations
            image = self.transforms(image=image)['image']

        if self.model == 'retinanet':
            return self.yolo_to_retinanet(idx, image, boxes, labels)

        return image

    def __len__(self):
        """
        Retorna o número de amostras no dataset.

        Returns:
            int: Número de imagens no dataset.
        """
        return len(self.images)
