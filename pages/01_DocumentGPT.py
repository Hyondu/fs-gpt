import streamlit as st

from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough


st.set_page_config(
  page_title="DocumentGPT",
  page_icon="📑"
)

chat = ChatOllama(model="mistral:latest")


@st.cache_data(show_spinner="Embedding the file...")
def embed_file(file):
  file_content = file.read()
  file_path = f"./.cache/files/{file.name}"
  with open(file_path, "wb") as f:
    f.write(file_content)
  cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
  splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
    )
  loader = UnstructuredFileLoader(file_path)
  docs = loader.load_and_split(text_splitter=splitter)
  embeddings = OllamaEmbeddings(model="mistral:latest")
  cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
      embeddings, cache_dir)
  db = FAISS.from_documents(docs, cached_embeddings)
  retriever = db.as_retriever()
  return retriever


def send_message(msg, role, save=True):
  with st.chat_message(role):
    st.markdown(msg)
  if save:
    st.session_state["messages"].append({"message": msg, "role": role})


def paint_history():
  for msg in st.session_state["messages"]:
    send_message(msg["message"], msg["role"], save=False)


def format_docs(documents):
  return "\n\n".join(doc.page_content for doc in documents)


prompt = ChatPromptTemplate.from_messages([
  ("system", """
   Answer the question using ONLY the following context. If you don't know the answer, just say you don't know.
   DON'T make anything up. Answer in Korean.

   Context: {context}
   """),
  ("human", "{question}"),
])

st.title("DocumentGPT")

st.markdown("""
Welcome!

Use this chatbot to ask questions to an AI about your files!

Upload your file on the sidebar.
""")

with st.sidebar:
  file = st.file_uploader("upload a .txt .pdf or .docx file", type=["txt", "pdf", "docx"])

if file:
  retriever = embed_file(file)
  send_message("I'm ready! Ask away!", "ai", save=False)
  paint_history()
  msg = st.chat_input("Ask anything about your file...")
  if msg:
    send_message(msg, "human")
    chain = {
      "context": retriever | RunnableLambda(format_docs),
      "question": RunnablePassthrough(),
      } | prompt | chat
    resp = chain.invoke(msg)
    send_message(resp.content, "ai")
else:
  st.session_state["messages"] = []