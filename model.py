from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import openvino_genai as ov_genai
import os

from config import EMBED_MODEL

tei_embedding_endpoint = os.getenv("TEI_ENDPOINT")

def load_knowledgeBase():
        if tei_embedding_endpoint:
            # create embeddings using TEI endpoint service
            embedder = HuggingFaceEndpointEmbeddings(model=tei_embedding_endpoint)
        else:
            # create embeddings using local embedding model
            embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embedder, allow_dangerous_deserialization=True)
        return db

knowledgeBase = load_knowledgeBase()

prompt_template_without_context = PromptTemplate.from_template(
    """You are a helpful assistant

Question:
{question}

Answer:"""
)

prompt_template_with_context = PromptTemplate.from_template(
    """You are a helpful assistant. Use only the below context and answer the question. Remove any unecessary information

Context:
{context}

Question:
{question}

Answer:"""
)

pipe_llama = ov_genai.LLMPipeline("Llama-3.2-3B-Instruct", "CPU")
pipe_qwen = ov_genai.LLMPipeline("Qwen2.5-3B-Instruct", "CPU")
pipe_phi = ov_genai.LLMPipeline("Phi-3.5-mini-instruct-fp16-ov", "CPU")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_text(class_range: str, query: str):
    class_range_parts = class_range.split('-')
    start_grade = int(class_range_parts[0].split()[1])
    end_grade = int(class_range_parts[1])
    grade_list = list(range(start_grade, end_grade + 1))

    prompt_without_context = prompt_template_without_context.format(question=query)

    result_llama = pipe_llama.generate(prompt_without_context, max_new_tokens=100, do_sample=False)
    result_qwen = pipe_qwen.generate(prompt_without_context, max_new_tokens=100, do_sample=False)
    result_phi = pipe_phi.generate(prompt_without_context, max_new_tokens=100, do_sample=False)

    similar_docs = knowledgeBase.similarity_search(query, k=5)
    filtered_docs = [
        doc for doc in similar_docs if int(doc.metadata['grade']) in grade_list
    ] 
    context = format_docs(filtered_docs)
    print(context)
    prompt_with_context = prompt_template_with_context.format(context=context, question=query)

    result_llama_rag = pipe_llama.generate(prompt_with_context, max_new_tokens=100, do_sample=False)
    result_qwen_rag = pipe_qwen.generate(prompt_with_context, max_new_tokens=100, do_sample=False)
    result_phi_rag = pipe_phi.generate(prompt_with_context, max_new_tokens=100, do_sample=False)
    
    return {
        "answer_llama": result_llama,
        "answer_qwen": result_qwen,
        "answer_phi": result_phi,
        "answer_llama_rag": result_llama_rag,
        "answer_qwen_rag": result_qwen_rag,
        "answer_phi_rag": result_phi_rag
    }