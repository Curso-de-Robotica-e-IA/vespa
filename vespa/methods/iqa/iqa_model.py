from abc import ABC, abstractmethod

from torch.nn import Module

class IQABaseModel(ABC, Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self, batch_size: int, epochs: int):
        """Train the model on the provided dataset."""
        pass

    @abstractmethod
    def valid(self, batch_size: int):
        """Validate the model on the provided dataset."""
        pass

    @abstractmethod
    def test(self, batch_size: int):
        """Evaluate the model on the test dataset."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Performs inferences on the model given an unlabeled dataset."""
        pass

    @abstractmethod
    def load(self, model_path: str):
        """Load the model from a file with pretrained weights."""
        pass
