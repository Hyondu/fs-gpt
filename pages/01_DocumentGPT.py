import os
import shutil
from operator import itemgetter
import streamlit as st

from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.globals import set_debug, set_verbose


set_debug(False)
set_verbose(False)


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📑"
)


with st.sidebar:
  model = st.selectbox("Choose LLM model",
                       ("mistral:latest", "llama2:latest", "llama2:13b", "llama3"))
  temperature = st.slider("Temperature", 0.1, 1.0)
  file = st.file_uploader("upload a .txt .pdf or .docx file", type=[
                          "txt", "pdf", "docx"])


st.title("문서GPT")

st.markdown("""
어서오세요!

당신의 문서에 관해 AI 챗봇에게 물어보세요!

왼쪽 사이드바에서 문서를 업로드 해보세요.
""")


def send_message(msg, role, save=True):
  with st.chat_message(role):
    st.markdown(msg)
  if save:
    save_massage(msg, role)


def save_massage(msg, role):
  st.session_state.messages.append({"message": msg, "role": role})


def save_memory(input, output):
  st.session_state.memory.save_context({"input": input}, {"output": output})


def invoke_chain(chain, msg):
  res = chain.invoke(msg)
  save_memory(msg, res.content)


def paint_history():
  for msg in st.session_state["messages"]:
    send_message(msg["message"], msg["role"], save=False)


def format_docs(documents):
  return "\n\n".join(doc.page_content for doc in documents)


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
  embeddings = OllamaEmbeddings(model=model)
  cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
      embeddings, cache_dir)
  db = FAISS.from_documents(docs, cached_embeddings)
  retriever = db.as_retriever()
  return retriever


class ChatCallbackHandler(BaseCallbackHandler):
  def __init__(self):
    self.message = ""
    self.message_box = None

  def on_llm_start(self, *args, **kwargs):
    self.message_box = st.empty()
    with st.sidebar:
      st.write("llm started")

  def on_llm_end(self, *args, **kwargs):
    save_massage(self.message, "ai")
    with st.sidebar:
      st.write("llm ended")

  def on_llm_new_token(self, token, *args, **kwargs):
    self.message += token
    self.message_box.markdown(self.message)


chat = ChatOllama(model=model, temperature=temperature,
                  callbacks=[ChatCallbackHandler()])

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     주어진 문서의 내용만을 활용해서 질문에 답해주세요. 만일 답을 모르면 모른다고 하세요.
     절대 아무 대답이나 지어내지 마세요. 한국어로 대화해 주세요.

     문서: {context}
     """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


if file:
  retriever = embed_file(file)
  send_message("준비돼써!! 무엇이든 물어보세요!!", "ai", save=False)
  paint_history()
  msg = st.chat_input("업로드한 문서에 관해 무엇이든 물어봐요.")
  if msg:
    send_message(msg, "human")
    chain = {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough.assign(
        history=RunnableLambda(
            st.session_state.memory.load_memory_variables
        ) | itemgetter("history")
    ) | prompt | chat
    with st.chat_message("ai"):
      invoke_chain(chain, msg)
else:
  try:
    shutil.rmtree("./.cache/files")
    shutil.rmtree("./.cache/embeddings")
    os.makedirs("./.cache/embeddings")
    os.makedirs("./.cache/files")
  except (OSError, FileNotFoundError):
    pass
  st.session_state["messages"] = []
  st.session_state["memory"] = ConversationSummaryBufferMemory(
      llm=chat, max_token_limit=2000, return_messages=True
  )
