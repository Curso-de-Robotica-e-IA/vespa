import os
import cv2

from vespa.datasets.base_dataset import BaseDataset
from tqdm import tqdm

from vespa.datasets.config import MEAN_YOLO, STD_YOLO
from vespa.datasets.yolo.yolo_transforms import get_yolo_test_transforms

class YOLODataset(BaseDataset):
    def __init__(self, root_dir, txt_file, image_size, transforms=None, model: str=None, train_mode = True):
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
        self.train_mode = train_mode

        # Lê o arquivo txt com as imagens e labels
        with open(self.txt_file_path) as f:
            self.images = f.read().strip().split("\n")

            self.labels = [label.replace('.jpg', '.txt')
                           .replace('/images/', '/labels/')
                            for label in self.images]
        
        
        self.verify_images()
        self.verify_labels()
    
    def verify_labels(self):
        before_size = len(self.labels)
        confirms = [os.path.normpath(os.path.join(self.root_dir, label))
                    for label in tqdm(self.labels)
                    if os.path.isfile(os.path.normpath(os.path.join(self.root_dir, label)))]
        self.labels = confirms
        
        labels_len = len(self.labels)
        if labels_len == 0:
            raise(Exception(f'No labels found from paths in {self.txt_file_path}'))
        
        print(f'{labels_len} labels read from {before_size}')

    
    def verify_images(self):
        before_size = self.__len__()
        confirms = [os.path.normpath(os.path.join(self.root_dir, image)) 
                    for image in tqdm(self.images) 
                    if os.path.isfile(os.path.normpath(os.path.join(self.root_dir, image)))]
        self.images = confirms

        if self.__len__() == 0:
            raise(Exception(f'No images found from paths in {self.txt_file_path}'))
        
        print(f'{self.__len__()} images read from {before_size}')

    def __getitem__(self, idx):
        """
        Retorna uma amostra do dataset no formato esperado pelo PyTorch.

        Args:
            idx (int): Índice do item.

        Returns:
            tuple: Imagem transformada e dicionário com alvos
                   (caixas e rótulos).
        """
        img = cv2.imread(self.images[idx])
        if img is None:
            raise FileNotFoundError(f'Image not found: {self.images[idx]}')

        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Construa o caminho do rótulo
        label_path = self.labels[idx]
        if not os.path.exists(label_path):
            raise FileNotFoundError(f'Label not found: {label_path}')

        if self.model == 'retinanet':
            return self.yolo_to_retinanet(idx, img, label_path, self.transforms, self.train_mode)

        return img

    def __len__(self):
        """
        Retorna o número de amostras no dataset.

        Returns:
            int: Número de imagens no dataset.
        """
        return len(self.images)

if __name__ == '__main__':
    d = YOLODataset(root_dir=r'\\192.168.155.240\Robotica\CME\dataset_cme_v4\laparoscopia_06-2024\tools - v2',
                    txt_file='train.txt', image_size=640,
                    transforms=get_yolo_test_transforms(),
                    model='retinanet',
                    train_mode=False)
    print(d[3])
