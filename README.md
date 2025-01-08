# VESPA
Computer Vision Python module for training and testing models.

## CUDA initial Setup

To use CUDA with Pytorch, you need to install the CUDA Toolkit.

1. Download local file CUDA Toolkit 12.1 or higher from [here](https://developer.nvidia.com/cuda-downloads).
2. Run the executable file and follow the instructions.

## Poetry Installation

To install poetry you can follow the instructions from official documentation [here](https://python-poetry.org/docs/). 

## Dependencies Installation

To install the dependencies, run the following command:

```bash
poetry install
```

## Using environment

To use the environment, run the following command:

```bash
poetry shell
```

### Check the CUDA availability

To check if CUDA is available, run the following command:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the output is `True`, CUDA is available.