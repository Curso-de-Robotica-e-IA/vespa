import pytest
from unittest.mock import MagicMock, patch
from vespa.methods.detection.onnx_yolo.model import ONNX_YOLO

@pytest.fixture
def dummy_model():
    """Cria uma instância da classe ONNX_YOLO com mock."""
    classes = {0: "class_0", 1: "class_1"}
    with patch("onnxruntime.InferenceSession") as mock_session:
        mock_session.return_value.get_inputs.return_value = [MagicMock(shape=[1, 3, 640, 640])]
        mock_session.return_value.get_outputs.return_value = [MagicMock(name="output")]
        model = ONNX_YOLO(model_path="dummy.onnx", classes=classes)
    return model