import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def create_vectorstore_from_chunks(
    chunks: list[Document],
    embedding_model_name: str = "BAAI/bge-m3",
    persist_path: str = "./data/faiss_index"
) -> None:
    """
    ساخت بردارساز و ذخیره FAISS از لیست چانک‌ها.
    """
    print(f"🧠 در حال ساخت embedding با مدل: {embedding_model_name}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = FAISS.from_documents(chunks, embedding_model)

    print(f"💾 ذخیره پایگاه برداری در مسیر: {persist_path}")
    if not os.path.exists(os.path.dirname(persist_path)):
        os.makedirs(os.path.dirname(persist_path))

    vector_store.save_local(persist_path)
    print("✅ ذخیره موفقیت‌آمیز انجام شد.")
