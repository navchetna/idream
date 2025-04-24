from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os

from config import EMBED_MODEL

tei_embedding_endpoint = os.getenv("TEI_ENDPOINT")

if tei_embedding_endpoint:
    # create embeddings using TEI endpoint service
    embedder = HuggingFaceEndpointEmbeddings(model=tei_embedding_endpoint)
else:
    # create embeddings using local embedding model
    embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

def count_documents(path, embedder):
    vectorstore = FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)
    return len(vectorstore.index_to_docstore_id)

before_count = count_documents("vectorstore/db_faiss", embedder)
print(before_count)