import streamlit as st
from langchain.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖",
)

st.title("FullstackGPT Home")

st.subheader("Your streamlit application here.")

st.markdown("""
  # Hello!
  
  Welcome to my FullstackGPT Portfolio!

  Here are the apps I made:

  - [ ] [DocumentGPT](/DocumentGPT)
  - [ ] [PrivateGPT](/PrivateGPT)
  - [ ] [QuizGPT](/QuizGPT)
  - [ ] [SiteGPT](/SiteGPT)
  - [ ] [MeetingGPT](/MeetingGPT)
  - [ ] [InvestorGPT](/InvestorGPT)
  """)

# Tabs in a single page
# tab1, tab2, tab3 = st.tabs(["A", "B", "C"])

# with tab1:
#   st.write("a")
#   st.markdown("""
#               # Chapter 1
#               ## part 1
#               """)

#   st.write(ChatPromptTemplate)

#   model = st.selectbox("Choose your model", ["gpt3", "gpt4", "mistral"])

#   st.write("The model you selected: ", model)
