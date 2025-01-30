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


# TODO: Fix this test case with correct dataset
def test_model_trainable_cpu(retina_sketch_fixture, create_dataset_path_train):
    root_dir = create_dataset_path_train

    dataset = COCODataset(
        root_dir=root_dir,
        txt_file=os.path.join(root_dir, 'train.txt'),
        transforms=None,
    )

    model = retina_sketch_fixture
    model.train()
    model.fit(train_dataset=dataset, batch_size=1, epochs=1, device='cpu')

    assert model.training is True
