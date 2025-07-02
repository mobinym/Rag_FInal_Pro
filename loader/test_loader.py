from loader.docx_loader import load_and_clean_docx

file_path = r"C:\Users\m.yaghoubi\Desktop\rag_phaz_project\data\docs\adamiyan_[www.ketabesabz.com].docx"  # فایل تستت اینجا بذار
docs = load_and_clean_docx(file_path)

print("📃 تعداد سندها:", len(docs))
print("🧹 پیش‌نمایش محتوا:\n", docs[0].page_content[:1000])
