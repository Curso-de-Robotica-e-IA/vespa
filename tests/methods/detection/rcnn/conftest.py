import pytest

from vespa.methods.detection.rcnn.model import RCNN


@pytest.fixture
def rcnn_pretrained_fixture():
    return RCNN(num_classes=5)


@pytest.fixture
def rcnn_sketch_fixture():
    return RCNN(num_classes=5, weights=None)
