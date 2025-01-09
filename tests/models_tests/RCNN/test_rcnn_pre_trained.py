import pytest
import torch
from vespa.methods.rcnn.model import RCNN


@pytest.fixture
def rcnn_model():
    """
    Fixture to initialize the RCNN model for testing.
    """
    return RCNN(num_classes=9)


def test_rcnn_model_exists(rcnn_model):
    """
    Test that the RCNN model is initialized correctly.
    """
    assert rcnn_model is not None, "RCNN model failed to initialize."


def test_rcnn_forward_pass_output_type(rcnn_model):
    """
    Test that the forward pass of the RCNN model returns a list.
    """
    dummy_input = torch.rand(2, 3, 256, 256)
    outputs = rcnn_model(dummy_input)
    assert isinstance(outputs, list), "RCNN forward pass did not return a list."


def test_rcnn_forward_pass_output_length(rcnn_model):
    """
    Test that the output list length matches the batch size.
    """
    dummy_input = torch.rand(2, 3, 256, 256)
    outputs = rcnn_model(dummy_input)
    assert len(outputs) == 2, f"Expected 2 outputs, but got {len(outputs)}."


def test_rcnn_output_contains_boxes(rcnn_model):
    """
    Test that the RCNN output contains the 'boxes' key.
    """
    dummy_input = torch.rand(2, 3, 256, 256)
    outputs = rcnn_model(dummy_input)
    assert "boxes" in outputs[0], "Output missing 'boxes' key."


def test_rcnn_output_contains_labels(rcnn_model):
    """
    Test that the RCNN output contains the 'labels' key.
    """
    dummy_input = torch.rand(2, 3, 256, 256)
    outputs = rcnn_model(dummy_input)
    assert "labels" in outputs[0], "Output missing 'labels' key."


def test_rcnn_output_contains_scores(rcnn_model):
    """
    Test that the RCNN output contains the 'scores' key.
    """
    dummy_input = torch.rand(2, 3, 256, 256)
    outputs = rcnn_model(dummy_input)
    assert "scores" in outputs[0], "Output missing 'scores' key."


def test_rcnn_model_without_pretrained_weights():
    """
    Test initializing the RCNN model without pretrained weights.
    """
    model = RCNN(num_classes=9)
    assert model is not None, "RCNN model without pretrained weights failed to initialize."
