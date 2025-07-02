from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def build_qa_chain(retriever, model_name: str = "gemma3"):
    llm = Ollama(model=model_name)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
         فقط بر اساس متن زیر به سؤال پاسخ بده و از هیچ دانش بیرونی استفاده نکن و حتما فارسی باشه اگر هم سوال نامربوط بود محترمانه بگو در سند موجود نیست :

        متن:
        {context}

        سؤال:
        {question}

        پاسخ:
        """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
