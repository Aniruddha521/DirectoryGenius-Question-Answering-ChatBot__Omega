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


def text_bar_cleaner():
    if 'input' not in st.session_state:
        st.session_state['input'] = ""
    if 'query' not in st.session_state:
        st.session_state['query'] = ""
    st.session_state["query"] = st.session_state["input"]
    st.session_state.update(input="")

def display_output_in_steamlit(text, delay=1):
    """
    Display the given text in markdown format with a specified delay,
    clearing the previous content before displaying the new text.

    Parameters:
    - text: str, The text to be displayed.
    - delay: float, The delay in seconds before displaying the text.
    """
    accumulated_text = ''
    placeholder = st.empty()
    for word in text.split(' '):
        accumulated_text += f"{word} "
        placeholder.markdown(accumulated_text)
        time.sleep(delay)
        st.empty()

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


if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
if 'query' not in st.session_state:
        st.session_state['query'] = ""

chat_style = """
                    <style>
                    .user-message-container {
                                            display: flex;
                                            justify-content: flex-end;
                                            width: 100%;
                                        }
                    .user-message {
                                        background-color:  #fdfefe;
                                        padding: 10px;
                                        border-radius: 15px;
                                        color: #00796b;
                                        margin-bottom: 10px;
                                        text-align: right;
                                        display: inline-block;
                                        max-width: 45%;
                                        min-width: 10%;
                                        word-wrap: break-word;
                                    }
                    </style>
                    """
col1, col2 = st.columns([10, 1])  # Adjust ratios as needed

with col2:
    button_clicked = st.button("Ask", key="ask_button")

with col1:
    st.text_input("", key="input", placeholder="Ask Omega.....",on_change=text_bar_cleaner)

st.markdown("""
<style>
.stButton > button {
    font-size: 36px; /* Increase font size of the button */
    width: 100%; /* Full width of its column */
    height: 40px; /* Set height to match the input field */
    margin: 27px 0 0 0; /* Top margin of 27px, other margins are 0 */    border-radius: 10px; /* Optional: add border-radius for aesthetics */
    background-color:  #FF5733 ;  /* Optional: change button background color */
    color: white;  /* Optional: change button text color */
    cursor: pointer;  /* Change cursor on hover for button */
}

.stButton > button:hover {
    background-color:  #bf360c ;  /* Change color on hover */
    color: white;
}

.stTextInput > div > div > input {
    width: 100%; /* Full width of its column */
    height: 40px; /* Set height to match the button */
    margin: 0; /* Ensure no extra margin */
    vertical-align: middle; /* Aligns vertically */

}
</style>
""", unsafe_allow_html=True)

if "chat_IDs" not in st.session_state:
    st.session_state["chat_IDs"] = False
if not st.session_state["chat_IDs"]:
    if not os.path.exists("chat_IDs.json"):
        with open("chat_IDs.json", "w") as file:
            json.dump({"details": []}, file, indent = 4)
    else:
        unique_id1 = uuid.uuid1()
        with open("chat_IDs.json", 'r') as file:
            id = json.load(file)
        id["details"].append({st.session_state['query']: str(unique_id1)})
        with open("chat_IDs.json", "w") as file:
                json.dump(id, file, indent = 4)
if st.session_state['query']:
    st.markdown(f"<div class='user-message-container'><div class='user-message'><strong>{st.session_state['query']}</strong></div></div>", unsafe_allow_html=True)
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
    reply = qa_chain({"query": st.session_state['query']} )
    st.session_state.conversation.append([st.session_state['query'] , reply["result"]])
    display_output_in_steamlit(reply["result"], delay=0.05)


st.markdown(chat_style, unsafe_allow_html=True)

# Display the conversation history with flexbox and custom styles
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for user_message, bot_message in st.session_state.conversation[0:-1][::-1]:
        st.markdown(f"<div class='user-message-container'><div class='user-message'><strong>{user_message}</strong></div></div>", unsafe_allow_html=True)
        st.markdown(bot_message)
    # else:
    #     st.markdown(f"<div class='bot-message-container'><div class='bot-message'><strong>Bot:</strong> {message}</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
try:
    with open(f"memory/{unique_id1}.json", "w") as file:
            json.dump(st.session_state.conversation, file, indent = 4)
except:
    st.session_state.update(query="")