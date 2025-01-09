import torch
import yaml
import os

# Normalização de imagens
MEAN = (0.485, 0.456, 0.406)  # Média de normalização (ImagemNet padrão)
STD = (0.229, 0.224, 0.225)   # Desvio padrão de normalização (ImagemNet padrão)

# Parâmetros de treinamento
DEFAULT_IMAGE_SIZE = 1088      # Tamanho padrão para redimensionamento das imagens
BATCH_SIZE = 16                # Tamanho do batch
NUM_WORKERS = 4                # Número de workers para o DataLoader
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Dispositivo para treinamento

# Caminhos padrões
DATASET_ROOT = "./vespa/datasets"    # Diretório raiz para datasets
OUTPUT_DIR = "./vespa/datasets/outputs"       # Diretório para salvar saídas (modelos, logs, etc.)
HYPERPARAMS_FILE = "./vespa/datasets/hyps.yaml"  # Caminho para o arquivo consolidado de hyperparams

# Limiar de confiança para inferência
CONFIDENCE_THRESHOLD = 0.5     # Limiar de confiança para detecção de objetos
NMS_THRESHOLD = 0.4            # Limiar para supressão não máxima (NMS)


def load_hyperparams(dataset_name: str) -> dict:
    """
    Carrega os hiperparâmetros do arquivo hyps.yaml para o dataset especificado.

    Args:
        dataset_name (str): Nome do dataset (ex.: "YOLO", "COCO").

    Returns:
        dict: Dicionário de hiperparâmetros para o dataset.

    Raises:
        ValueError: Se o dataset não for encontrado no hyps.yaml.
    """
    if not os.path.exists(HYPERPARAMS_FILE):
        raise FileNotFoundError(f"Arquivo de hiperparâmetros '{HYPERPARAMS_FILE}' não encontrado.")

    with open(HYPERPARAMS_FILE, "r") as file:
        hyperparams = yaml.safe_load(file)

    datasets = hyperparams.get("Datasets", {})
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' não encontrado no hyps.yaml.")

    return datasets[dataset_name]


def save_hyperparams(hyperparams: dict):
    """
    Salva os hiperparâmetros no arquivo hyps.yaml.

    Args:
        hyperparams (dict): Dicionário consolidado de hiperparâmetros.
    """
    with open(HYPERPARAMS_FILE, "w") as file:
        yaml.dump({"Datasets": hyperparams}, file, default_flow_style=False)
        print(f"Hiperparâmetros salvos em '{HYPERPARAMS_FILE}'")
