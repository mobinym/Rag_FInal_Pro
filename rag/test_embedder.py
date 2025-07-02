import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loader.docx_loader import load_and_clean_docx
from rag.chunker import split_documents_into_chunks
from rag.embedder import create_vectorstore_from_chunks

file_path = "./data/docs/adamiyan_[www.ketabesabz.com].docx"

print(f"📄 در حال بارگذاری فایل: {file_path}")
docs = load_and_clean_docx(file_path)
chunks = split_documents_into_chunks(docs)

create_vectorstore_from_chunks(
    chunks=chunks,
    embedding_model_name="BAAI/bge-m3",
    persist_path="./data/faiss_index"
)
