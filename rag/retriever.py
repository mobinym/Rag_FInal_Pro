from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_retriever(
    persist_path: str = "./data/faiss_index",
    embedding_model_name: str = "BAAI/bge-m3",
    top_k: int = 5
):
    """
    بارگذاری VectorStore و ساخت یک بازیاب ساده از روی آن.
    """
    print(f"📁 Loading FAISS index from: {persist_path}")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vector_store = FAISS.load_local(
        persist_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True  # فقط در محیط لوکال استفاده شود
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    print("✅ Retriever loaded successfully.")
    return retriever
