import pytest

from vespa.methods.detection.retinanet.model import RetinaNet


@pytest.fixture
def retina_pretrained_fixture():
    return RetinaNet(num_classes=5)


@pytest.fixture
def retina_sketch_fixture() -> RetinaNet:
    return RetinaNet(num_classes=5, weights=None)
