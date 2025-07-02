import time
import logging
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from rag.retriever import build_retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer

# پیکربندی لاگر
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# شمارنده توکن خروجی
def get_token_count(text: str, model_name: str = "BAAI/bge-m3") -> int:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    return len(tokens)

# اجرای پرسش و پاسخ
def run_query(qa_chain, query: str):
    logging.info(f"Query: {query}")
    start_time = time.time()

    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    duration = round(time.time() - start_time, 2)

    logging.info(f"Answer: {answer}")
    logging.info(f"Token Count: {get_token_count(answer)}")
    logging.info(f"Duration: {duration} seconds")
    print("🧠 Answer:", answer)
    print("-" * 50)

# تابع اصلی
def main():
    retriever = build_retriever()
    llm = OllamaLLM(model="gemma3")  # اگر خطا داشت، بررسی کن که مدل gemma3 نصب باشه

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    while True:
        query = input("Enter your question (type 'exit' to quit):\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        run_query(qa_chain, query)

if __name__ == "__main__":
    main()
