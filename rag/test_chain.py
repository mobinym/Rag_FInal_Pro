import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.retriever import load_retriever_from_local_faiss
from rag.chain import build_rag_chain

retriever = load_retriever_from_local_faiss()
rag_chain = build_rag_chain(retriever, llm_model_name="gemma3")

print("🔁 System is ready. Type your questions below.")
print("Type 'exit' to quit.")
print("-" * 50)

while True:
    query = input("You: ")
    if query.strip().lower() in ["exit", "خروج"]:
        break

    print("⏳ Thinking...")
    answer = rag_chain.invoke(query)
    
    print("\n🧠 Answer:")
    print(answer)
    print("-" * 50)
