from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import device, DB_FAISS_PATH, EMBEDDING_TYPE, DATA_PATH, EMBEDDING_FOLDER
import os

# create vector database
def create_vector_db():
    modelPath = EMBEDDING_FOLDER + EMBEDDING_TYPE

    if not os.path.exists(DB_FAISS_PATH):
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                                    chunk_overlap = 50,
                                                    separators = ["\n\n", "\n", " ", "","."])

        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name = modelPath,
                                        model_kwargs = {'device':device})

        db = FAISS.from_documents(texts, embeddings)
        db.save_local(DB_FAISS_PATH)
        print("Done")
    else:
        print("Folder already exists.")

if __name__ == '__main__':
    create_vector_db()
