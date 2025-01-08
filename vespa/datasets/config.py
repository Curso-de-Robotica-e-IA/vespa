import torch

# Normalização de imagens
MEAN = (0.485, 0.456, 0.406)  # Média de normalização (ImagemNet padrão)
STD = (0.229, 0.224, 0.225)   # Desvio padrão de normalização (ImagemNet padrão)

# Parâmetros de treinamento
DEFAULT_IMAGE_SIZE = 512       # Tamanho padrão para redimensionamento das imagens
BATCH_SIZE = 16                # Tamanho do batch
NUM_WORKERS = 4                # Número de workers para o DataLoader
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Dispositivo para treinamento

# Caminhos padrões
DATASET_ROOT = "./datasets"    # Diretório raiz para datasets
OUTPUT_DIR = "./outputs"       # Diretório para salvar saídas (modelos, logs, etc.)

# Limiar de confiança para inferência
CONFIDENCE_THRESHOLD = 0.5     # Limiar de confiança para detecção de objetos
NMS_THRESHOLD = 0.4            # Limiar para supressão não máxima (NMS)
