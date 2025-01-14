import pytest
from vespa.methods.retinanet.model import RetinaNet

@pytest.fixture
def retina_pretrained_fixture():
    return RetinaNet()

@pytest.fixture
def retina_sketch_fixture():
    return RetinaNet(pre_trained=False)