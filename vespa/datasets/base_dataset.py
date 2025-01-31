from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from vespa.datasets.dataset_utils import DatasetUtils


class BaseDataset(Dataset, ABC, DatasetUtils):
    """
    Classe base abstrata para diferentes formatos de datasets.
    """

    def __init__(self, root_dir, transforms=None, model=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.model = model

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
