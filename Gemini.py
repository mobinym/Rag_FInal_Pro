# rag_system_v7_self_correcting_complete.py

import os
import re
import csv
import time
from datetime import datetime
from typing import List, Any
import numpy as np

# --- وارد کردن کتابخانه‌های اصلی ---
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# --- وارد کردن کامپوننت‌های پیشرفته برای Retriever ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from sentence_transformers import CrossEncoder

# ==============================================================================
# بخش ۱: تنظیمات پروژه
# ==============================================================================

DEBUG_MODE = True
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
LOCAL_LLM_NAME = "gemma3"

BASE_RETRIEVER_K = 10
FINAL_RETRIEVED_K = 3

CHUNK_SIZE = 300
CHUNK_OVERLAP = 60
DEFAULT_DOCUMENT_PATH = r"C:\path\to\your\document.docx" # لطفا مسیر فایل خود را اینجا قرار دهید
FEEDBACK_FILE_PATH = "feedback_logs/feedback.csv"

# ==============================================================================
# بخش ۲: تعریف کلاس‌ها و توابع کاربردی
# ==============================================================================

class RerankCompressor(BaseDocumentCompressor):
    """یک فشرده‌ساز سند سفارشی که از یک Cross-Encoder برای رتبه‌بندی مجدد اسناد استفاده می‌کند."""
    class Config:
        arbitrary_types_allowed = True

    reranker: CrossEncoder
    top_n: int = FINAL_RETRIEVED_K

    def compress_documents(self, documents: List[Document], query: str, **kwargs: Any) -> List[Document]:
        if not documents:
            return []
        
        doc_list = [doc.page_content for doc in documents]
        pairs = [[query, doc] for doc in doc_list]
        
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        
        docs_with_scores = list(zip(documents, scores))
        sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        
        result = []
        for doc, score in sorted_docs[:self.top_n]:
            doc.metadata['relevance_score'] = score
            result.append(doc)
        return result

def load_and_clean_document(file_path: str) -> list[Document] | None:
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return None
    print("1. Loading and cleaning the document...")
    try:
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        documents[0].page_content = documents[0].page_content.strip()
        print("Document loaded and cleaned successfully.")
        return documents
    except Exception as e:
        print(f"An error occurred while loading or cleaning the file: {e}")
        return None

def save_feedback(report_data: dict, feedback: str, comment: str = "", correction: str = ""):
    os.makedirs(os.path.dirname(FEEDBACK_FILE_PATH), exist_ok=True)
    headers = [
        "timestamp", "question", "retrieved_context", "retrieval_scores",
        "raw_answer", "feedback_status", "user_comment", "corrected_answer",
        "retrieval_time_sec", "generation_time_sec"
    ]
    file_exists = os.path.isfile(FEEDBACK_FILE_PATH)
    try:
        with open(FEEDBACK_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(headers)
            scores = [f"{doc.metadata.get('relevance_score', 0):.4f}" for doc in report_data.get("retrieved_docs", [])]
            writer.writerow([
                datetime.now().isoformat(), report_data.get("question", ""),
                report_data.get("context_str", ""), ", ".join(scores),
                report_data.get("answer", ""), feedback, comment, correction,
                f"{report_data.get('retrieval_time', 0):.4f}", f"{report_data.get('generation_time', 0):.4f}"
            ])
        if feedback: print("Feedback saved successfully. Thank you!")
    except IOError as e:
        print(f"Error writing to feedback file: {e}")

# ==============================================================================
# بخش ۳: هسته اصلی سیستم RAG
# ==============================================================================

class RAGCore:
    def __init__(self, file_path: str):
        self.advanced_retriever = None
        self.llm = None
        self.prompt_template = None
        self._initialize_pipeline(file_path)

    def _initialize_pipeline(self, file_path: str):
        documents = load_and_clean_document(file_path)
        if not documents:
            raise FileNotFoundError(f"Document file not found at '{file_path}'.")
        
        print("2. Splitting the document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(documents)

        print("3. Creating embeddings and vector store...")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
        vector_store = FAISS.from_documents(chunks, embedding_model)
        print("Vector Store created successfully.")

        print("4. Setting up Advanced Retriever with Re-Ranking...")
        base_retriever = vector_store.as_retriever(search_kwargs={"k": BASE_RETRIEVER_K})
        
        reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device='cpu')
        compressor = RerankCompressor(reranker=reranker_model, top_n=FINAL_RETRIEVED_K)
        
        self.advanced_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        print("Advanced Retriever is ready.")
        
        self.llm = OllamaLLM(model=LOCAL_LLM_NAME, temperature=0.0)
        
        prompt_str = """### نقش ###
شما یک تحلیل‌گر داده متخصص و یک حقیقت‌یاب (Fact-Checker) بسیار دقیق هستید.
### هدف اصلی ###
هدف اصلی شما پاسخ به سوال کاربر با دقت فوق‌العاده بالا است. برای این کار، شما باید متن «زمینه» را به دقت درک کرده، ارتباطات منطقی ساده (مانند نسبت دادن ضمیر "من" به گوینده متن) را برقرار کنی، و پاسخی دقیق فقط و فقط بر اساس اطلاعات موجود در زمینه ارائه دهی.
### قوانین اصلی (بایدها) ###
1.  **استخراج کلمه به کلمه:** پاسخ را تا حد امکان کلمه به کلمه از متن استخراج کن.
2.  **تحلیل جامع زمینه:** به تمام جزئیات (فاعل‌ها، مفعول‌ها، تاریخ‌ها، اعداد، نام‌های خاص) دقت کن.
3.  **پاسخ مستقیم و خلاصه:** فقط به سوال پرسیده شده پاسخ بده. از کپی کردن تمام پاراگراف خودداری کن.
4.  **زبان پاسخ:** پاسخ خود را فقط و فقط به زبان فارسی بنویس.
### محدودیت‌های کلیدی (نبایدها) ###
1.  **ممنوعیت دانش خارجی:** از دانش قبلی خود استفاده نکن.
2.  **ممنوعیت استنتاج پیچیده:** اطلاعاتی را که به صراحت در متن بیان نشده، حدس نزن.
3.  **مدیریت پاسخ ناموجود:** اگر پاسخ در «زمینه» یافت نمی‌شود، باید دقیقاً با عبارت «پاسخ این سوال در سند موجود نیست.» جواب دهی.
---
زمینه:
{context}
---
سوال:
{question}
---
پاسخ دقیق و مبتنی بر متن:
"""
        self.prompt_template = PromptTemplate(template=prompt_str, input_variables=["context", "question"])
        print("5. LLM and Prompt are initialized.")

    def get_response_and_report(self, question: str) -> dict:
        start_time_retrieval = time.time()
        retrieved_docs = self.advanced_retriever.invoke(question)
        end_time_retrieval = time.time()
        
        context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        final_prompt = self.prompt_template.format(context=context_str, question=question)
        
        start_time_generation = time.time()
        answer = self.llm.invoke(final_prompt)
        end_time_generation = time.time()
        
        return {
            "question": question, "retrieved_docs": retrieved_docs, "context_str": context_str,
            "final_prompt": final_prompt, "answer": answer,
            "retrieval_time": end_time_retrieval - start_time_retrieval,
            "generation_time": end_time_generation - start_time_generation,
        }

    def self_correct_answer(self, report_data: dict) -> dict:
        """یک پاسخ را که فیدبک 'بد' گرفته، بازبینی و در صورت امکان اصلاح می‌کند."""
        print("\n🔄 Self-correction loop initiated...")
        context_str = report_data["context_str"]
        original_answer = report_data["answer"]

        critique_prompt_str = """متن زیر را به عنوان «زمینه» و ادعای پس از آن را به عنوان «ادعا» در نظر بگیر.
آیا «ادعا» به طور کامل و دقیق توسط «زمینه» پشتیبانی می‌شود؟ فقط با 'بله' یا 'خیر' پاسخ بده.

زمینه:
{context}
---
ادعا:
{claim}
---
پاسخ (بله/خیر):
"""
        critique_prompt = PromptTemplate(template=critique_prompt_str, input_variables=["context", "claim"])
        critique_chain = critique_prompt | self.llm
        critique_result = critique_chain.invoke({"context": context_str, "claim": original_answer}).strip().lower()
        
        print(f"Self-critique result: LLM believes the answer was supported? -> '{critique_result}'")

        if "خیر" in critique_result or "no" in critique_result:
            print("Diagnosis: Generation Failure detected. Attempting to re-generate...")
            correction_prompt = report_data["final_prompt"] + "\n\nیادآوری: پاسخ قبلی شما توسط زمینه پشتیبانی نمی‌شد. لطفاً دوباره با دقت بسیار بیشتری فقط بر اساس زمینه پاسخ دهید."
            new_answer = self.llm.invoke(correction_prompt)
            return {"corrected": True, "new_answer": new_answer, "reason": "Original answer was not supported by the context."}
        else:
            print("Diagnosis: Retrieval Failure suspected. Cannot improve with current context.")
            return {"corrected": False, "new_answer": "متاسفانه با اطلاعات موجود در سند، قادر به ارائه پاسخ بهتری نیستم.", "reason": "The original answer seems correct based on the provided context, which itself might be irrelevant."}

# ==============================================================================
# بخش ۴: حلقه اصلی برنامه
# ==============================================================================

def print_debug_report(report: dict):
    print("\n" + "="*25 + " DEBUG REPORT " + "="*25)
    print(f"\n[1. QUESTION]: {report['question']}")
    print(f"\n[2. ADVANCED RETRIEVER PERFORMANCE]:")
    print(f"   - Time taken: {report['retrieval_time']:.4f} seconds")
    print(f"   - Chunks retrieved and re-ranked: {len(report['retrieved_docs'])}")
    print("\n[3. FINAL RETRIEVED CHUNKS (Post Re-Ranking)]:")
    for i, doc in enumerate(report['retrieved_docs']):
        score = doc.metadata.get('relevance_score', 'N/A')
        score_str = f"{score:.4f}" if isinstance(score, float) else score
        print(f"   --- Chunk {i+1} (Re-Ranker Score: {score_str}) ---")
        indented_content = "      " + doc.page_content.replace("\n", "\n      ")
        print(indented_content)
    print(f"\n[4. LLM PERFORMANCE]:")
    print(f"   - Time taken: {report['generation_time']:.4f} seconds")
    print(f"\n[5. LLM FINAL ANSWER]:")
    print(f"   -> {report['answer']}")
    print("\n" + "="*24 + " END OF REPORT " + "="*25 + "\n")

def main():
    print("=" * 50)
    print("Welcome to the Intelligent Document Q&A System (v7.0 - Self-Correcting)")
    print("=" * 50)

    user_doc_path = input(f"Please enter the full path to the .docx file (or press Enter for default):\n[{DEFAULT_DOCUMENT_PATH}]\n> ")
    if not user_doc_path:
        user_doc_path = DEFAULT_DOCUMENT_PATH

    try:
        rag_system = RAGCore(file_path=user_doc_path)
    except Exception as e:
        print(f"\nAn unexpected error occurred during system startup: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ System is fully initialized and ready.")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 50)

    while True:
        user_question = input("You: ")
        if user_question.lower() in ["exit", "quit", "خروج"]:
            print("Goodbye!")
            break
        
        report_data = rag_system.get_response_and_report(user_question)
        
        if DEBUG_MODE:
            print_debug_report(report_data)
        
        print("\nSystem's Answer:")
        print(report_data.get("answer", "No answer was received."))
        print("-" * 20)

        feedback_input = input("Was this answer helpful? (1: Good / 2: Bad / Enter: Skip)\n> ").lower()
        
        if feedback_input in ["2", "bad"]:
            print("\nI understand you weren't satisfied. Let me review and correct the answer...")
            correction_result = rag_system.self_correct_answer(report_data)
            
            print("\n✅ Corrected Answer:")
            print(correction_result["new_answer"])
            print(f"(Reason for correction: {correction_result['reason']})")
            
            feedback_status = "bad_then_corrected" if correction_result["corrected"] else "bad_uncorrectable"
            save_feedback(report_data, feedback=feedback_status, correction=correction_result["new_answer"])

        elif feedback_input in ["1", "good"]:
            save_feedback(report_data, feedback="good")
        
        print("-" * 50)

if __name__ == "__main__":
    main()