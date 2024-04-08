import streamlit as st

st.title("DocumentGPT")

with st.chat_message("human"):
  st.write("Hello!")

with st.chat_message("ai"):
  st.write("How are you?")

st.chat_input("Send a message to the AI")


with st.status("Embedding file...", expanded=True) as status:
  import time
  time.sleep(2)
  st.write("Getting the file")
  time.sleep(2)
  st.write("Embedding the file")
  time.sleep(2)
  st.write("Caching the file")
  status.update(label="Error", state="error")


def send_message(message, role):
  with st.chat_message(role):
    st.write(message)
