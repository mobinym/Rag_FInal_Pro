from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
import re
import os

def load_and_clean_docx(file_path: str) -> list[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ فایل '{file_path}' پیدا نشد.")

    print(f"📄 در حال بارگذاری فایل: {file_path}")
    loader = Docx2txtLoader(file_path)
    documents = loader.load()

    raw_text = documents[0].page_content

    cleaned_text = re.sub(r'www\.ketabesabz\.com', '', raw_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'صفحه[:\s]*\d+', '', cleaned_text)
    cleaned_text = re.sub(r'آدمیان \d+ ‎\d+‏', '', cleaned_text)
    cleaned_text = re.sub(r'\d+ #? ?زویا قلی‌پور', '', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)

    return [Document(page_content=cleaned_text)]
