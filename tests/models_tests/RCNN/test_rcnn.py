import torch
from torch import Tensor
from torch.utils.data import DataLoader
from vespa.methods.rcnn.model import RCNN


def test_model_list(rcnn_pretrained_fixture, tensor_image_fixture):
    model = rcnn_pretrained_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    assert isinstance(results, list)


def test_model_dict(rcnn_pretrained_fixture, tensor_image_fixture):
    model = rcnn_pretrained_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result, dict)


def test_model_boxes(rcnn_pretrained_fixture, tensor_image_fixture):
    model = rcnn_pretrained_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result['boxes'], Tensor)


def test_model_scores(rcnn_pretrained_fixture, tensor_image_fixture):
    model = rcnn_pretrained_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result['scores'], Tensor)


def test_model_labels(rcnn_pretrained_fixture, tensor_image_fixture):
    model = rcnn_pretrained_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result['labels'], Tensor)


def test_rcnn_pretrained_initialization(rcnn_pretrained_fixture):
    model = rcnn_pretrained_fixture
    assert hasattr(model.model, "roi_heads"), "Pretrained RCNN model should have an ROI head"


def test_rcnn_sketch_initialization(rcnn_sketch_fixture):
    model = rcnn_sketch_fixture
    assert hasattr(model.model, "roi_heads"), "Custom RCNN model should have an ROI head"


def test_rcnn_optimizer_configuration():
    model = RCNN(optimizer_name = 'sgd')
    model.configure_optimizer()
    assert isinstance(model.optimizer, torch.optim.SGD), "Optimizer should be SGD when configured"
