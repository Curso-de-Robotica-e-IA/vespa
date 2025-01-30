import pytest

from vespa.methods.retinanet.model import RetinaNet


@pytest.fixture
def retina_pretrained_fixture():
    return RetinaNet()


@pytest.fixture
def retina_sketch_fixture() -> RetinaNet:
    return RetinaNet(weights=None)
