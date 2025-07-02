from loader.docx_loader import load_and_clean_docx
from rag.chunker import split_documents_into_chunks
import rag.chunker
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("🧠 Chunker loaded from:", rag.chunker.__file__)
print("🔍 dir(chunker):", dir(rag.chunker))

file_path = r"C:\Users\m.yaghoubi\Desktop\rag_phaz_project\data\docs\adamiyan_[www.ketabesabz.com].docx"
docs = load_and_clean_docx(file_path)
chunks = split_documents_into_chunks(docs)

print("📚 تعداد چانک‌ها:", len(chunks))
print("📄 پیش‌نمایش چانک اول:\n", chunks[0].page_content)
