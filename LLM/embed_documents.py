from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

"""

embed_documents.py serves to vectorize chunked data and save it to a specifc path.

TODO: add functionality to dynamically append more data to existing vectorized data

"""

# create path to save vectorized chunk data
index_path = "data/faiss_index"

def embed_chunks(chunks):
    # initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # wrap each chunk as a Document
    documents = [Document(page_content=chunk) for chunk in chunks]

    # vectorize chunks
    vectorstore = FAISS.from_documents(documents, embedding_model)

    # create the directory if it does not already exist
    os.makedirs(index_path, exist_ok=True)

    # save the vectorized data to the set path
    vectorstore.save_local(index_path)