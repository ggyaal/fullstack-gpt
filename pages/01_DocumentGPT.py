import os
import streamlit as st

from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

@st.cache_data(show_spinner="Embedding files...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path=file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

prompt = ChatPromptTemplate.from_messages([
    ("system", """
Answer the question using ONLY the context provided below.  
If the answer isn’t in the context, just say you don’t know—don’t try to make something up.  
Respond in Korean using casual and friendly language, like how close friends talk.
  
Context: {context}
    """),
    ("human", "{question}")
])

st.title("📄 DocumentGPT")

with st.sidebar:
    api_key = st.text_input(
        label="OPEN AI API KEY",
        placeholder="enter your OPEN AI API KEY!"
    )

    if api_key:
      file = st.file_uploader(
          "Upload a .txt .pdf or .docx file",
          type=["pdf", "txt", "docx"],
      )

      os.environ["OPENAI_API_KEY"] = api_key

      llm = ChatOpenAI(
          temperature=0.1,
          streaming=True,
          callbacks=[
              ChatCallbackHandler(),
          ],
      )

if api_key and file:
    retriever = embed_file(file)
    send_message("오 읽어봤어! 이제 질문해도 돼!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []
    st.markdown("""
**DocumentGPT는 파일을 업로드하면 내용을 임베딩하여, 사용자의 질문에 정확히 답해주는 문서 기반 AI 챗봇입니다.**

PDF, TXT 등 다양한 형식의 문서를 불러오면, AI가 문서를 분석하고 벡터로 임베딩합니다. 이후 사용자의 질문에 대해 **문서 내용을 기반으로 직접적인 답변**을 제공하므로, 설명서, 보고서, 논문, 매뉴얼 등 어떤 문서에도 유용하게 활용할 수 있습니다.

---

## ✨ 주요 기능

- 📁 문서 업로드 및 자동 임베딩 처리
- 💬 문서 기반 질의응답 (QA)
- 🧠 GPT 기반 자연어 이해 및 응답
- 🔍 문서 내용을 바탕으로 정밀하고 근거 있는 답변 제공
- 🗂 다중 문서 지원 (`*.txt`, `*.pdf`, `*.docx`)

---
    """)
