import os
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Optional, Union
from fastapi import File, Form, HTTPException, UploadFile
from pathlib import Path

from comps import CustomLogger, DocPath, opea_microservices, register_microservice
from comps.parsers.treeparser import TreeParser
from comps.parsers.tree import Tree
from comps.parsers.node import Node
from comps.parsers.text import Text
from comps.parsers.table import Table
from config import EMBED_MODEL
from utils import (
    create_upload_folder,
    get_separators,
    encode_filename
)

logger = CustomLogger("ncert_dataprep")
logflag = os.getenv("LOGFLAG", False)

tei_embedding_endpoint = os.getenv("TEI_ENDPOINT")
upload_folder = "./uploaded_files/"
tree_parser = TreeParser()

async def save_content_to_local_disk(save_path: str, content):
    save_path = Path(save_path)
    try:
        if isinstance(content, str):
            with open(save_path, "w", encoding="utf-8") as file:
                file.write(content)
        else:
            with save_path.open("wb") as fout:
                content = await content.read()
                fout.write(content)
    except Exception as e:
        print(f"Write file failed. Exception: {e}")
        raise Exception(status_code=500, detail=f"Write file {save_path} failed. Exception: {e}")

def ingest_chunks_to_faiss(chunks: List, text_splitter: RecursiveCharacterTextSplitter, grade: str, subject: str):

    DB_FAISS_PATH = 'vectorstore/db_faiss'

    # Create vectorstore
    if tei_embedding_endpoint:
        # create embeddings using TEI endpoint service
        embedder = HuggingFaceEndpointEmbeddings(model=tei_embedding_endpoint)
    else:
        # create embeddings using local embedding model
        embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

    metadata = {"grade": grade, "subject": subject}
    metadata_list = [metadata] * len(chunks)

    documents = text_splitter.create_documents(texts = chunks, metadatas = metadata_list)

    if os.path.exists(DB_FAISS_PATH):
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embedder, allow_dangerous_deserialization=True)
        vectorstore.add_documents(documents)
    else:
        vectorstore = FAISS.from_documents(documents=documents, embedding=embedder)

    vectorstore.save_local(DB_FAISS_PATH)

    return True

def get_table_description(item: Table):
    server_host_ip = os.getenv("LLM_SERVER_HOST_IP")
    server_port = os.getenv("LLM_SERVER_PORT")
    url = f"http://{server_host_ip}:{server_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    data = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": """
                    <s>[INST] <<SYS>>\n You are a helpful, respectful, and honest assistant. Your task is to generate a detailed and descriptive summary of the provided table data in Markdown format, based strictly on the table and its heading. <</SYS>> 
                    [INST] Your job is to create a clear, specific, and **factual** textual description. **Do not add any external information** or provide an abstract summary. Only base the description on the data from the table and its heading.
                    
                    1. Link the **columns** with the corresponding **values** in the rows, referencing the exact terms and terminology from the table. 
                    2. For each row, explain how each column's data relates to the corresponding values. Ensure the description is **step-by-step** and follows the structure of the table in a natural order.
                    3. **Do not return the table itself.** Provide only the descriptive summary, written in **paragraphs**.
                    4. The description should be precise, direct, and **avoid interpretation** or generalization. Stay true to the exact data given.
                    
                    Think carefully and make sure to describe every column and its respective values in detail. 
                """
            },
            {
                "role": "user",
                "content": f"{item.heading}\n{item.markdown_content}",
            }
        ],
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data)
    response_data = json.loads(response.text)
    return response_data['choices'][0]['message']['content']

def chunk_node_content(node: Node, text_splitter: RecursiveCharacterTextSplitter):
    content = node.get_content()
    chunks = []
    for item in content:
        if isinstance(item, Text):
            text_chunks = text_splitter.split_text(item.content)
            chunks.extend(text_chunks)
        if isinstance(item, Table):
            table_description = get_table_description(item)
            table_description_chunks = text_splitter.split_text(table_description)
            chunks.extend(table_description_chunks)
    return chunks

def create_chunks(node: Node, text_splitter: RecursiveCharacterTextSplitter):
    node_chunks = chunk_node_content(node, text_splitter)
    total = node.get_length_children()
    for i in range(total):
        node_chunks.extend(create_chunks(node.get_child(i), text_splitter))
    return node_chunks

def ingest_data_to_faiss(doc_path: DocPath, grade: str, subject: str):
    """Ingest document to FAISS."""
    path = doc_path.path
    if logflag:
        logger.info(f"[ ingest data ] Parsing document {path}.")

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=doc_path.chunk_size,
            chunk_overlap=doc_path.chunk_overlap,
            add_start_index=True,
            separators=get_separators(),
        )

    tree = Tree(path)
    tree_parser = TreeParser()
    tree_parser.populate_tree(tree)
    # return
    chunks = create_chunks(tree.rootNode, text_splitter)

    return ingest_chunks_to_faiss(chunks, text_splitter, grade, subject)

@register_microservice(name="opea_service@prepare_doc_faiss", endpoint="/v1/dataprep", host="0.0.0.0", port=6007)
async def ingest_documents(
    files: Optional[Union[UploadFile, List[UploadFile]]] = File(None),
    link_list: Optional[str] = Form(None),
    chunk_size: int = Form(300),
    chunk_overlap: int = Form(60),
    process_table: bool = Form(False),
    table_strategy: str = Form("fast"),
    grade: int = Form(...),
    subject: str = Form(...), 
):
    if logflag:
        logger.info(f"[ upload ] files:{files}")
        logger.info(f"[ upload ] link_list:{link_list}")

    if files:
        if not isinstance(files, list):
            files = [files]
        uploaded_files = []

        for file in files:
            encode_file = encode_filename(file.filename)
            doc_id = "file:" + encode_file
            if logflag:
                logger.info(f"[ upload ] processing file {doc_id}")

            save_path = upload_folder + encode_file
            await save_content_to_local_disk(save_path, file)
            ingest_data_to_faiss(
                DocPath(
                    path=save_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    process_table=process_table,
                    table_strategy=table_strategy,
                ),
                grade,
                subject
            )
            uploaded_files.append(save_path)
            if logflag:
                logger.info(f"[ upload ] Successfully saved file {save_path}")

        result = {"status": 200, "message": "Data preparation succeeded"}
        if logflag:
            logger.info(result)
        return result

    raise HTTPException(status_code=400, detail="Must provide either a file or a string list.")

if __name__ == "__main__":
    create_upload_folder(upload_folder)
    opea_microservices["opea_service@prepare_doc_faiss"].start()