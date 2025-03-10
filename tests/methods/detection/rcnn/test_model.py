import torch
from torch import Tensor
import torch.nn.functional as F


def test_model_list(rcnn_pretrained_fixture, tensor_image_fixture):
    rcnn_pretrained_fixture.eval()
    results = rcnn_pretrained_fixture(tensor_image_fixture)

    assert isinstance(results, list)


def test_model_dict(rcnn_pretrained_fixture, tensor_image_fixture):
    rcnn_pretrained_fixture.eval()
    results = rcnn_pretrained_fixture(tensor_image_fixture)

    for result in results:
        assert isinstance(result, dict)


def test_model_boxes(rcnn_pretrained_fixture, tensor_image_fixture):
    rcnn_pretrained_fixture.eval()
    results = rcnn_pretrained_fixture(tensor_image_fixture)

    for result in results:
        assert isinstance(result['boxes'], Tensor)


def test_model_scores(rcnn_pretrained_fixture, tensor_image_fixture):
    rcnn_pretrained_fixture.eval()
    results = rcnn_pretrained_fixture(tensor_image_fixture)

    for result in results:
        assert isinstance(result['scores'], Tensor)


def test_model_labels(rcnn_pretrained_fixture, tensor_image_fixture):
    rcnn_pretrained_fixture.eval()
    results = rcnn_pretrained_fixture(tensor_image_fixture)

    for result in results:
        assert isinstance(result['labels'], Tensor)


def test_model_train_loop_cpu(rcnn_pretrained_fixture, yolo_dataset_train_rcnn):
    initial_weights = {name: param.clone() for name, param in rcnn_pretrained_fixture.named_parameters()}
    
    rcnn_pretrained_fixture.fit(yolo_dataset_train_rcnn, 1, 1, 'cpu')

    for name, param in rcnn_pretrained_fixture.named_parameters():
        assert not torch.equal(initial_weights[name], param), f"Peso {name} não mudou após o treino!"


def test_model_sketch_list(rcnn_sketch_fixture, tensor_image_fixture):
    rcnn_sketch_fixture.eval()
    results = rcnn_sketch_fixture(tensor_image_fixture)

    assert isinstance(results, list)


def test_model_sketch_dict(rcnn_sketch_fixture, tensor_image_fixture):
    rcnn_sketch_fixture.eval()
    results = rcnn_sketch_fixture(tensor_image_fixture)

    for result in results:
        assert isinstance(result, dict)


def test_model_sketch_boxes(rcnn_sketch_fixture, tensor_image_fixture):
    rcnn_sketch_fixture.eval()
    results = rcnn_sketch_fixture(tensor_image_fixture)

    for result in results:
        assert isinstance(result['boxes'], Tensor)


def test_model_sketch_scores(rcnn_sketch_fixture, tensor_image_fixture):
    rcnn_sketch_fixture.eval()
    results = rcnn_sketch_fixture(tensor_image_fixture)

    for result in results:
        assert isinstance(result['scores'], Tensor)


def test_model_sketch_labels(rcnn_sketch_fixture, tensor_image_fixture):
    rcnn_sketch_fixture.eval()
    results = rcnn_sketch_fixture(tensor_image_fixture)

    for result in results:
        assert isinstance(result['labels'], Tensor)


def test_model_sketch_train_loop_gpu(
    rcnn_sketch_fixture, yolo_dataset_train_rcnn
):
    rcnn_sketch_fixture.fit(yolo_dataset_train_rcnn, 4, 4, 0)


def test_model_sketch_train_loop_cpu(
    rcnn_sketch_fixture, yolo_dataset_train_rcnn
):
    rcnn_sketch_fixture.fit(yolo_dataset_train_rcnn, 1, 1, 'cpu')



def test_model_boxes_random_noise(rcnn_pretrained_fixture):
    """Testa o modelo com uma imagem de ruído aleatório e verifica se retorna detecções."""
    rcnn_pretrained_fixture.eval()

    input_tensor = torch.rand((3, 32, 32))

    with torch.no_grad():
        result = rcnn_pretrained_fixture([input_tensor])

    assert isinstance(result[0]['boxes'], Tensor)


def test_model_labels_random_noise(rcnn_pretrained_fixture):
    """Testa o modelo com uma imagem de ruído aleatório e verifica se retorna detecções."""
    rcnn_pretrained_fixture.eval()

    input_tensor = torch.rand((3, 32, 32))

    with torch.no_grad():
        result = rcnn_pretrained_fixture([input_tensor])

    assert isinstance(result[0]['labels'], Tensor)


def test_model_trainable(rcnn_pretrained_fixture):
    """Verifica se o modelo pode ser treinado sem erro em um batch pequeno."""
    rcnn_pretrained_fixture.train()

    input_tensor = torch.rand((3, 32, 32))
    target = [{
        "boxes": torch.tensor([[5, 5, 15, 15]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64)
    }]

    loss_dict = rcnn_pretrained_fixture([input_tensor], target)
    loss = sum(loss for loss in loss_dict.values())

    assert loss.item() > 0, "A loss deve ser maior que 0 durante o treinamento"


def test_model_freeze_layers(rcnn_pretrained_fixture):
    """Verifica se o congelamento das camadas do backbone funciona corretamente."""
    for param in rcnn_pretrained_fixture.model.backbone.parameters():
        param.requires_grad = False

    frozen_params = [p.requires_grad for p in rcnn_pretrained_fixture.model.backbone.parameters()]
    
    assert not any(frozen_params), "Todas as camadas do backbone deveriam estar congeladas"