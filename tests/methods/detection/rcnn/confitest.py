import pytest

from vespa.methods.detection.rcnn.model import RCNN


@pytest.fixture
def rcnn_pretrained_fixture():
    return RCNN()


@pytest.fixture
def rcnn_sketch_fixture():
    return RCNN(pre_trained=False)
