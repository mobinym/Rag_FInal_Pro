from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents_into_chunks(
    documents: list[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 60,
    model_name: str = "gpt-4"
) -> list[Document]:
    """
    تقسیم لیست Documentها به چانک‌های کوچک‌تر.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
