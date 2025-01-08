import pytest
import torch

@pytest.fixture
def tensor_image_fixture():
    rgb = torch.randn(1, 3, 600, 600)
    return rgb