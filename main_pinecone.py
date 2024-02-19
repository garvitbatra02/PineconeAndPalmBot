import streamlit as st
from langchain_helper_pinecone import get_qa_chain

st.title("Garvit Batra Q/A bot🌱")
# btn = st.button("Create Knowledgebase")
# if btn:
#     create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])