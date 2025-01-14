from abc import ABC, abstractmethod

from torch.nn import Module
from vespa.methods.model_utils import ModelUtils


class BaseModel(ABC, Module, ModelUtils):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Perform the forward pass."""
        pass

    @abstractmethod
    def train(self, train_dataset, batch_size: int, epochs: int, device: str):
        """Train the model on the provided dataset."""
        pass

    @abstractmethod
    def valid(self, val_dataset, batch_size: int, device: str):
        """Validate the model and compute metrics."""
        pass

    @abstractmethod
    def test(self, test_dataset, batch_size: int, device: str):
        """Evaluate the model on the test dataset."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Performs inferences on the model given an unlabeled dataset."""
        pass
    