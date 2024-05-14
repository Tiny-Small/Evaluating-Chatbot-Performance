# config.py

import torch

# Configuration for device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DB_FAISS_PATH = "db_faiss"
LLM_PATH = "LLM/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf"
EMBEDDING_TYPE = "avsolatorio/GIST-large-Embedding-v0"
DATA_PATH = "data/"
EMBEDDING_FOLDER = "embedding_model/"
