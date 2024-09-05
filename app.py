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
from tkinter import Tk
from tkinter.filedialog import askdirectory
import base64
import uuid
import json
import time
import random
# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def text_bar_cleaner():
    if 'input' not in st.session_state:
        st.session_state['input'] = ""
    if 'query' not in st.session_state:
        st.session_state['query'] = ""
    st.session_state["query"] = st.session_state["input"]
    st.session_state.update(input="")

with open("giphy.gif", "rb") as gif_file:
    gif_data = gif_file.read()
    encoded_gif = base64.b64encode(gif_data).decode("utf-8")

st.markdown('<h1 class="centered-header">Directory Genius Chatbot</h1>', unsafe_allow_html=True)

# Display chat messages from history on app rerun
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
# Display assistant response in chat message container
with st.chat_message("assistant"):
    response = st.write_stream(response_generator())
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    # st.components.v1.html(f"{f.read()}", height=max, width=max, scrolling=True)
# # st.sidebar.header("Chat History")
# st.markdown(
#     f"""
#     <style>
#     [data-testid="stSidebar"] {{
#         background-image: url("data:image/gif;base64,{encoded_gif}");
#         background-size: cover;
#         background-position: center;
#     }}
    
#     [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .css-1avcm0n {{
#         color: white;
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
with st.sidebar:
    # st.markdown('<a href="#" class="sidebar-button">Styled Button</a>', unsafe_allow_html=True)
    selected_option = st.selectbox( " ",["Database 1", "Database 2", "Database 3"], placeholder="Select DataBase", key="database_selection", help="Select database to retrive information")
    st.header("Chat History")
    st.write("Click the styled button!")
# with st.sidebar:
#     
#     st.write("This is a modified sidebar!")
    # st.image("https://media.giphy.com/media/26tn33aiTi1jkl6H6/giphy.gif")
