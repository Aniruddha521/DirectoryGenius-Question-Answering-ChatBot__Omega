# from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI, ChatGooglePalm, ChatVertexAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
# from langchain.agents import AgentExecutor
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from tools import retrival_question_answering
import streamlit as st
from utilities import *
from prompts import *
import uuid
import json
import time

def display_with_clear_and_delay(text, delay=1):
    """
    Display the given text in markdown format with a specified delay,
    clearing the previous content before displaying the new text.

    Parameters:
    - text: str, The text to be displayed.
    - delay: float, The delay in seconds before displaying the text.
    """
    placeholder = st.empty()  # Create a placeholder for the text
    time.sleep(delay)  # Add delay
    placeholder.markdown(text) 

folder_path = "/home/roy/Langchain/langchain"
databasename = folder_path.split("/")[-1]
set_API(name="chatgroq", platform="c")
set_API(name="huggingface", platform="hf")
set_API(name="serpapi", platform="s")
embedd_model = "all-mpnet-base-v2"
cache_folder_name = "cache/mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedd_model, cache_folder=cache_folder_name)
dict_ignore_files = {
    "dir": [".github", ".git", ".devcontainer", "docs", "examples", "sample_documents"],
    "extension": ["faiss", "pdf", "cff","png", "jpg", "gif", "yaml", "cff", "Dockerfile", "lock", "toml", "odt", "gitmodules", "csv", "example", "ambr", "html", "typed", "xml", "yml", "avi"],
    "file": [".gitignore", ".dockerignore", ".flake8", ".gitattributes", "Dockerfile", "LICENSE", "Makefile"]
}
db2 = DeepLake(dataset_path=f"Database/{databasename}-{cache_folder_name}", read_only=True, embedding_function=embeddings)
st.title("DirectoryGenius Chatbot Omega")
model2 = ChatGroq( model_name="Llama3-70b-8192")#Mixtral-8x7b-32768



question = st.text_input("Ask your Query")

if question:
    st.session_state.retriever = db2.as_retriever()
    st.session_state.retriever.search_kwargs['distance_metric'] = 'cos'
    st.session_state.retriever.search_kwargs['fetch_k'] = 100
    # retriever.search_kwargs['maximal_marginal_relevance'] = True
    st.session_state.retriever.search_kwargs['k'] = 20

    memory = ConversationBufferMemory(
        llm=model2,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=2000
    )

    qa_chain = RetrievalQA.from_chain_type(
        model2,
        retriever=st.session_state.retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    reply = qa_chain({"query": question} )
    display_with_clear_and_delay(reply["result"])