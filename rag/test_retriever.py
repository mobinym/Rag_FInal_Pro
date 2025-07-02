import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.retriever import load_retriever_from_local_faiss

retriever = load_retriever_from_local_faiss()

query = "در این کتاب چه موضوعی بررسی می‌شود؟"
results = retriever.get_relevant_documents(query)

print(f"🔎 Top {len(results)} relevant chunks:")
print("-" * 50)
for i, doc in enumerate(results, 1):
    print(f"Chunk {i}:\n{doc.page_content[:300]}")
    print("-" * 50)
