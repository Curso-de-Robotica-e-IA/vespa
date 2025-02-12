from torch import Tensor


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


def test_model_train_loop_gpu(
    rcnn_pretrained_fixture, yolo_dataset_train_rcnn
):
    model = rcnn_pretrained_fixture
    dataset = yolo_dataset_train_rcnn

    model.fit(dataset, 4, 4, 0)


def test_model_train_loop_cpu(
    rcnn_pretrained_fixture, yolo_dataset_train_rcnn
):
    model = rcnn_pretrained_fixture
    dataset = yolo_dataset_train_rcnn

    model.fit(dataset, 1, 1, 'cpu')


def test_model_sketch_list(rcnn_sketch_fixture, tensor_image_fixture):
    model = rcnn_sketch_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    assert isinstance(results, list)


def test_model_sketch_dict(rcnn_sketch_fixture, tensor_image_fixture):
    model = rcnn_sketch_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result, dict)


def test_model_sketch_boxes(rcnn_sketch_fixture, tensor_image_fixture):
    model = rcnn_sketch_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result['boxes'], Tensor)


def test_model_sketch_scores(rcnn_sketch_fixture, tensor_image_fixture):
    model = rcnn_sketch_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result['scores'], Tensor)


def test_model_sketch_labels(rcnn_sketch_fixture, tensor_image_fixture):
    model = rcnn_sketch_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result['labels'], Tensor)


def test_model_sketch_train_loop_gpu(
    rcnn_sketch_fixture, yolo_dataset_train_rcnn
):
    model = rcnn_sketch_fixture
    dataset = yolo_dataset_train_rcnn

    model.fit(dataset, 4, 4, 0)


def test_model_sketch_train_loop_cpu(
    rcnn_sketch_fixture, yolo_dataset_train_rcnn
):
    model = rcnn_sketch_fixture
    dataset = yolo_dataset_train_rcnn

    model.fit(dataset, 1, 1, 'cpu')
