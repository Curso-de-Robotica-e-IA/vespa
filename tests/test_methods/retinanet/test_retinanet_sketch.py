import os

from torch import Tensor

from vespa.datasets.coco.coco_dataset import COCODataset


def test_model_list(retina_sketch_fixture, tensor_image_fixture):
    model = retina_sketch_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    assert isinstance(results, list)


def test_model_dict(retina_sketch_fixture, tensor_image_fixture):
    model = retina_sketch_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result, dict)


def test_model_boxes(retina_sketch_fixture, tensor_image_fixture):
    model = retina_sketch_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result['boxes'], Tensor)


def test_model_scores(retina_sketch_fixture, tensor_image_fixture):
    model = retina_sketch_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result['scores'], Tensor)


def test_model_labels(retina_sketch_fixture, tensor_image_fixture):
    model = retina_sketch_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result['labels'], Tensor)


def test_model_train_loop_gpu(retina_pretrained_fixture, yolo_dataset_train_retina):
    model = retina_pretrained_fixture
    dataset = yolo_dataset_train_retina

    model.fit(dataset, 4, 4, 0)

def test_model_train_loop_cpu(retina_pretrained_fixture, yolo_dataset_train_retina):
    model = retina_pretrained_fixture
    dataset = yolo_dataset_train_retina

    model.fit(dataset, 1, 1, 'cpu')
