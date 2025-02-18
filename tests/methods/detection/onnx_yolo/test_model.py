import torch
from unittest.mock import patch

def test_model_initialization(dummy_model):
    """Testa a inicialização da classe ONNX_YOLO."""
    assert dummy_model.model_path == "dummy.onnx"

def test_model_classes(dummy_model):
    """Verifica se o dicionário de classes está correto."""
    assert dummy_model.classes == {0: "class_0", 1: "class_1"}

def test_confidence_threshold(dummy_model):
    """Verifica se o confidence threshold está correto."""
    assert dummy_model.confidence_threshold == 0.5

def test_iou_threshold(dummy_model):
    """Verifica se o IoU threshold está correto."""
    assert dummy_model.iou_threshold == 0.5

def test_input_height(dummy_model):
    """Verifica a altura da entrada do modelo."""
    assert dummy_model.input_height == 640

def test_input_width(dummy_model):
    """Verifica a largura da entrada do modelo."""
    assert dummy_model.input_width == 640

def test_model_predict(dummy_model):
    """Testa o método predict com uma entrada de tensor mockada."""
    with patch.object(dummy_model.session, "run", return_value=[{"boxes": [[10, 20, 30, 40]], "scores": [0.9], "labels": [1]}]):
        image = torch.rand(3, 640, 640)
        results = dummy_model.predict([image])
    assert len(results) == 1  # Deve retornar uma lista de detecções por imagem

def test_load_model(dummy_model):
    """Testa o carregamento de um novo modelo ONNX."""
    with patch("onnxruntime.InferenceSession") as mock_session:
        dummy_model.load("new_model.onnx")
        mock_session.assert_called_with("new_model.onnx", providers=["CPUExecutionProvider"])

def test_print_model_summary(dummy_model, capsys):
    """Testa se print_model_summary exibe as informações corretamente."""
    dummy_model.print_model_summary()
    captured = capsys.readouterr()
    assert "YOLO ONNX Model Path: dummy.onnx" in captured.out
