from torch import Tensor


def test_model_list(retina_pretrained_fixture, tensor_image_fixture):
    model = retina_pretrained_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    assert isinstance(results, list)

def test_model_dict(retina_pretrained_fixture, tensor_image_fixture):
    model = retina_pretrained_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result, dict)

def test_model_boxes(retina_pretrained_fixture, tensor_image_fixture):
    model = retina_pretrained_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result['boxes'], Tensor)

def test_model_scores(retina_pretrained_fixture, tensor_image_fixture):
    model = retina_pretrained_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result['scores'], Tensor)

def test_model_labels(retina_pretrained_fixture, tensor_image_fixture):
    model = retina_pretrained_fixture
    image = tensor_image_fixture
    model.eval()
    results = model(image)

    for result in results:
        assert isinstance(result['labels'], Tensor)

