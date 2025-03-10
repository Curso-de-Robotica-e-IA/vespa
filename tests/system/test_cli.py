import subprocess

import pytest


def run_cli_command(command: list):
    """
    Runs a CLI command and captures its stdout, stderr, and exit code.

    Args:
        command (list): The command to
                        run as a list (e.g., ["list-models"]).

    Returns:
        tuple: (stdout, stderr, exit_code)
    """

    result = subprocess.run(
        ['poetry', 'run', 'vespa'] + command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.stdout, result.stderr, result.returncode


def test_list_models_exit_code():
    """
    Test if the `list-models` command exits with code 0 (success).

    Fails if:
    - The command crashes due to an import error.
    - The `onnxruntime` package is missing.
    """
    _, stderr, exit_code = run_cli_command(['list-models'])
    assert exit_code == 0, (
        f'Expected exit code 0, but got {exit_code}. STDERR: {stderr}'
    )  # noqa


def test_list_models_output():
    """
    Test if the `list-models` command returns the expected output.

    Fails if:
    - The output does not contain 'Available Models:'.
    - The command crashes due to missing dependencies.
    """
    stdout, _, _ = run_cli_command(['list-models'])
    assert 'Available Models:' in stdout, (
        f"Expected 'Available Models:' in stdout, but got: {stdout}"
    )  # noqa


def test_list_models_no_stderr():
    """
    Test if the `list-models` command does not produce any error output.

    Fails if:
    - There is any content in `stderr` indicating an error.
    """
    _, stderr, _ = run_cli_command(['list-models'])
    assert 'Traceback' not in stderr, (
        f'Expected empty stderr, but got: {stderr}'
    )  # noqa


def test_list_dataset_formats_exit_code():
    """
    Test if the `list-dataset-formats` command exits with code 0.

    Fails if:
    - The command fails due to missing dependencies.
    """
    _, _, exit_code = run_cli_command(['list-dataset-formats'])
    assert exit_code == 0, f'Expected exit code 0, but got {exit_code}'


def test_list_dataset_formats_output():
    """
    Test if `list-dataset-formats` returns supported dataset formats.

    Fails if:
    - The output does not contain 'Supported Dataset Formats:'.
    """
    stdout, _, _ = run_cli_command(['list-dataset-formats'])
    assert 'Supported Dataset Formats:' in stdout, (
        f"Expected 'Supported Dataset Formats:' in stdout, but got: {stdout}"
    )  # noqa


def test_list_dataset_formats_no_stderr():
    """
    Test if `list-dataset-formats` does not produce error output.

    Fails if:
    - There is any content in `stderr` indicating an error.
    """
    _, stderr, _ = run_cli_command(['list-dataset-formats'])
    assert 'Traceback' not in stderr, (
        f'Expected empty stderr, but got: {stderr}'
    )  # noqa


def test_missing_train_params_stdout():
    """
    Test if the `train` command returns an
    error message when parameters are missing.

    Fails if:
    - The expected error message is not in `stderr`.
    """
    stdout, _, _ = run_cli_command(['train'])
    assert 'Error: Missing parameters for training.' in stdout, (
        f"Expected 'Error: Missing parameters for training.' in stderr, but got: {stdout}"  # noqa
    )


@pytest.mark.parametrize('model_name', ['retinanet'])
def test_train_model_exit_code(model_name):
    """
    Test if the `train` command exits successfully
    when all required parameters are provided.

    Parameters:
    - `model_name` (str): The model to be trained.

    Fails if:
    - The command exits with a non-zero code.
    """
    _, _, exit_code = run_cli_command([
        'train',
        '--task',
        'detection',
        '--model',
        model_name,
        '--format',
        'yolo',
        '--dataset',
        './assets/datasets/detection/yolo',
    ])
    assert exit_code == 0, (
        f'Expected exit code 0 for model {model_name}, but got {exit_code}'
    )  # noqa


@pytest.mark.parametrize('model_name', ['retinanet'])
def test_train_model_output(model_name):
    """
    Test if `train` prints a confirmation message when training starts.

    Parameters:
    - `model_name` (str): The model to be trained.

    Fails if:
    - The output does not contain 'Training Result:'.
    """
    stdout, _, _ = run_cli_command([
        'train',
        '--task',
        'detection',
        '--model',
        model_name,
        '--format',
        'yolo',
        '--dataset',
        './assets/datasets/detection/yolo',
    ])
    assert 'Training Result:' in stdout, (
        f"Expected 'Training Result:' in stdout, but got: {stdout}"
    )  # noqa


@pytest.mark.parametrize('model_name', ['retinanet'])
def test_predict_exit_code(model_name):
    """
    Test if `predict` exits successfully with valid parameters.

    Parameters:
    - `model_name` (str): The model used for prediction.

    Fails if:
    - The command exits with a non-zero code.
    """
    _, _, exit_code = run_cli_command([
        'predict',
        '--task',
        'detection',
        '--model',
        model_name,
        '--input',
        'assets/datasets/detection/yolo/test/images/image_0.jpg',
    ])
    assert exit_code == 0, (
        f'Expected exit code 0 for model {model_name}, but got {exit_code}'
    )  # noqa


@pytest.mark.parametrize('model_name', ['retinanet'])
def test_predict_output(model_name):
    """
    Test if `predict` produces the expected output format.

    Parameters:
    - `model_name` (str): The model used for prediction.

    Fails if:
    - The output does not contain 'Prediction Output:'.
    """
    stdout, _, _ = run_cli_command([
        'predict',
        '--task',
        'detection',
        '--model',
        model_name,
        '--input',
        'assets/datasets/detection/yolo/test/images/image_0.jpg',
    ])
    assert 'Prediction Output:' in stdout, (
        f"Expected 'Prediction Output:' in stdout, but got: {stdout}"
    )  # noqa
