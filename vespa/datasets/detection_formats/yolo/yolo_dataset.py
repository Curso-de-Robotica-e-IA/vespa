import os

import cv2
from albumentations import Compose
from tqdm import tqdm

from vespa.datasets.base_dataset import BaseDataset
from vespa.datasets.utils import yolo_to_retinanet, yolo_to_rcnn


class YOLODataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        txt_file: str,
        image_size: int = 416,
        transforms: Compose = None,
        model_name: str = None,
    ):
        """
        Initializes the YOLO dataset.
        Args:
            root_dir (str): The root directory where the dataset is stored.
            txt_file (str): The name of the text file containing image paths.
            image_size (int): The size to which images will be resized.
            transforms (Compose): The transformations to be applied to
            the images.
            model (str): The model type being used.
        Attributes:
            image_size (int): The size to which images will be resized.
            txt_file_path (str): The full path to the text file containing
            image paths.
            images (list): A list of image paths and labels read from the text
            file.
        Methods:
            verify_images(): Verifies the existence and validity of the images
            listed in the text file.
        """
        super().__init__(root_dir, transforms, model_name)
        self.image_size = image_size
        self.txt_file_path = os.path.join(root_dir, txt_file)

        # Lê o arquivo txt com as imagens e labels
        with open(self.txt_file_path, 'r', encoding='utf-8') as f:
            self.images = f.read().strip().split('\n')

        self.verify_images()

    def verify_images(self):
        """
        Verifies the existence of image and label files listed in the dataset.

        This method checks if the image files and their corresponding label
        files exist in the specified directories. It updates the list of
        images to only include those that have both image and label files
        present. If no valid images are found, it raises an exception.

        Raises:
            Exception: If no valid images are found in the dataset.
        """
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
        Retrieve the item at the specified index from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
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

        # Apply Albumentations transforms
        augmented = self.transforms(image=image, bboxes=boxes, labels=labels)
        image = augmented['image']
        boxes = augmented['bboxes']
        labels = augmented['labels']

        if self.model == 'retinanet':
            return yolo_to_retinanet(idx, image, boxes, labels)
        elif self.model == 'rcnn':
            return yolo_to_rcnn(idx, image, boxes, labels)
        elif self.model == 'yolo':
            return image, boxes, labels
        else:
            raise ValueError(f"Unsupported model type: {self.model}")

    def __len__(self):
        """
        Retorna o número de amostras no dataset.

        Returns:
            int: Número de imagens no dataset.
        """
        return len(self.images)
