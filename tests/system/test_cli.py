import subprocess
import pytest
import sys
import torch

CLI_PATH = "vespa/cli/cli.py"  # Adjust the path if necessary

def run_cli_command(args):
    """Helper function to run the CLI command and return output"""
    result = subprocess.run(
        ["python", CLI_PATH] + args, 
        capture_output=True, text=True
    )
    return result.stdout, result.stderr, result.returncode

def test_debug_python_path():
    print("Python Path:", sys.executable)
    print("Python Version:", sys.version)
    print("PyTorch Version:", torch.__version__)
    assert True  # Just a debug test

# ---- TEST LIST MODELS ----
def test_list_models_exit_code():
    stdout, stderr, exit_code = run_cli_command(["list-models"])
    print("STDOUT:", stdout)  # Debugging
    print("STDERR:", stderr)  # Debugging
    assert exit_code == 0  # This is failing


def test_list_models_output():
    stdout, stderr, _ = run_cli_command(["list-models"])
    print("STDOUT:", stdout)  # Debugging
    print("STDERR:", stderr)  # Debugging
    assert "Available Models:" in stdout

def test_list_models_no_stderr():
    _, stderr, _ = run_cli_command(["list-models"])
    assert stderr == ""

# ---- TEST LIST DATASET FORMATS ----
def test_list_dataset_formats_exit_code():
    _, _, exit_code = run_cli_command(["list-dataset-formats"])
    assert exit_code == 0

def test_list_dataset_formats_output():
    stdout, _, _ = run_cli_command(["list-dataset-formats"])
    assert "Supported Dataset Formats:" in stdout

def test_list_dataset_formats_no_stderr():
    _, stderr, _ = run_cli_command(["list-dataset-formats"])
    assert stderr == ""

# ---- TEST MISSING TRAIN PARAMS ----
def test_missing_train_params_exit_code():
    _, _, exit_code = run_cli_command(["train"])
    assert exit_code != 0  # Should return a non-zero error code

def test_missing_train_params_stderr():
    _, stderr, _ = run_cli_command(["train"])
    assert "Error: Missing parameters for training." in stderr

# ---- TEST TRAINING ----
def test_train_model_exit_code(root_path_dataset_yolo_format):
    _, _, exit_code = run_cli_command([
        "train", "--task", "detection", "--model", "retinanet",
        "--format", "yolo", "--dataset", root_path_dataset_yolo_format
    ])
    assert exit_code == 0

def test_train_model_output(root_path_dataset_yolo_format):
    stdout, _, _ = run_cli_command([
        "train", "--task", "detection", "--model", "retinanet",
        "--format", "yolo", "--dataset", root_path_dataset_yolo_format
    ])
    assert "Training Result:" in stdout

# ---- TEST PREDICTION ----
def test_predict_exit_code():
    _, _, exit_code = run_cli_command([
        "predict", "--task", "detection", "--model", "retinanet", 
        "--input", "assets/datasets/detection/yolo/test/images/image_0.jpg"
    ])
    assert exit_code == 0

def test_predict_output():
    stdout, _, _ = run_cli_command([
        "predict", "--task", "detection", "--model", "rcnn", 
        "--input", "assets/datasets/detection/yolo/test/images/image_0.jpg"
    ])
    assert "Prediction Output:" in stdout
