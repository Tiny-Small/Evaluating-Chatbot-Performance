import os
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_TYPE, EMBEDDING_FOLDER

# create vector database
def create_embedding_model():
    modelPath = EMBEDDING_FOLDER + EMBEDDING_TYPE

    if not os.path.exists(modelPath):
        model = SentenceTransformer(EMBEDDING_TYPE)
        model.save(modelPath)
        model = SentenceTransformer(modelPath)
    else:
        print("Folder already exists.")

if __name__ == '__main__':
    create_embedding_model()
