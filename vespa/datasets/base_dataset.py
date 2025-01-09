from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Classe base abstrata para diferentes formatos de datasets.
    """

    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

    @abstractmethod
    def __getitem__(self, idx):
        """
        Método obrigatório para carregar uma amostra do dataset.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Método obrigatório para retornar o tamanho do dataset.
        """
        pass
