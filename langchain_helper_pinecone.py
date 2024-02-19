import os
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain.embeddings import GooglePalmEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
import pinecone
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm=GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"],temperature=0)

# google_palm_embeddings = GooglePalmEmbeddings(google_api_key=os.environ["GOOGLE_API_KEY"])

# vector_file_path="faiss_index"
index={}
def create_vector_db():
    global index
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    pinecone.init(
        api_key="2f6bafd2-3f66-4326-9ce5-c927e306418e",
        environment="gcp-starter"
        )
    index_name="garrybotindex"
    index = Pinecone.from_existing_index(
                index_name,
                embedding=instructor_embeddings)
    
    
def get_qa_chain():
    # global index
    retriever = index.as_retriever(score_threshold = 0.7)

    prompt_template = """
    Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If answer is not found in the context, kindly state "I don't know."and let them know to contact me personally by sharing my linkedin(https://www.linkedin.com/in/garvit-batra-9b0a19205/) and github(https://github.com/garvitbatra02) handle present in response

    CONTEXT: {context}

    QUESTION: {question}"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    modified_chain = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=retriever,
                                input_key="query",
                                return_source_documents=True,
                                chain_type_kwargs=chain_type_kwargs)
    # print(modified_chain("Explain me how Garvit Batra would say Tell me something about yourself?"))
    return modified_chain

# if __name__ == "__main__":
    # create_vector_db()
    # chain=get_qa_chain()
    # print(chain("Tell me about coding skills of Garvit Batra"))
    

