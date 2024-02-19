import streamlit as st
from langchain_helper_pinecone import get_qa_chain, create_vector_db

st.title("Garvit Batra Q/A bot🌱")

# Button to create knowledge base
if st.button("Create Knowledgebase"):
    create_vector_db()

# Input for question
question = st.text_input("Question: ")

# Button to get answer
if st.button("Get Answer") and question:
    try:
        chain = get_qa_chain()
        response = chain(question)

        st.header("Answer")
        st.write(response["result"])
    except AttributeError as e:
        st.error("Please create the knowledge base first.")
