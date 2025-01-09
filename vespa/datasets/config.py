import os

import yaml

# Normalização de imagens
MEAN_COCO = (0.485, 0.456, 0.406)
STD_COCO = (0.229, 0.224, 0.225)

MEAN_YOLO = (0.485, 0.456, 0.406)
STD_YOLO = (0.229, 0.224, 0.225)

MEAN_PASCALVOC = (0.485, 0.456, 0.406)
STD_PASCALVOC = (0.229, 0.224, 0.225)


# Caminhos padrões
# Diretório raiz para datasets
DATASET_ROOT = './vespa/datasets'
# Diretório para salvar saídas (modelos, logs, etc.)
OUTPUT_DIR = './vespa/datasets/outputs'
# Caminho para o arquivo consolidado de hyperparams
HYPERPARAMS_FILE = './vespa/datasets/hyps.yaml'


def load_hyperparams(dataset_name: str) -> dict:
    """
    Carrega os hiperparâmetros do arquivo hyps.yaml
    para o dataset especificado.

    Args:
        dataset_name (str): Nome do dataset (ex.: "YOLO", "COCO").

    Returns:
        dict: Dicionário de hiperparâmetros para o dataset.

    Raises:
        ValueError: Se o dataset não for encontrado no hyps.yaml.
    """
    if not os.path.exists(HYPERPARAMS_FILE):
        raise FileNotFoundError(
            f"Arquivo de hiperparâmetros '{HYPERPARAMS_FILE}' não encontrado."
        )

    with open(HYPERPARAMS_FILE, 'r') as file:  # noqa
        hyperparams = yaml.safe_load(file)

    datasets = hyperparams.get('Datasets', {})
    if dataset_name not in datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' não encontrado no hyps.yaml."
        )

    return datasets[dataset_name]


def save_hyperparams(hyperparams: dict):
    """
    Salva os hiperparâmetros no arquivo hyps.yaml.

    Args:
        hyperparams (dict): Dicionário consolidado de hiperparâmetros.
    """
    with open(HYPERPARAMS_FILE, 'w') as file:  # noqa
        yaml.dump({'Datasets': hyperparams}, file, default_flow_style=False)
        print(f"Hiperparâmetros salvos em '{HYPERPARAMS_FILE}'")
