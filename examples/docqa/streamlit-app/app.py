import os

import streamlit as st
from utils import agent, configure

import langroid.language_models as lm
from langroid.utils.configuration import settings

settings.cache_type = "fakeredis"
if st.session_state.get("specified_file") is None:
    st.session_state["specified_file"] = ""
if st.session_state.get("file_path") is None:
    st.session_state["file_path"] = ""
if st.session_state.get("rag_agent") is None:
    st.session_state["rag_agent"] = None
if st.session_state.get("chat_model") is None:
    st.session_state["chat_model"] = None

default_chat_model = lm.OpenAIChatModel.GPT4o.value
chat_model = st.sidebar.text_input(
    f"""
Chat model, e.g. `litellm/ollama/mistral:7b-instruct-v0.2-q4_K_M`,
or leave empty to default to {default_chat_model}
"""
)
actual_chat_model = chat_model or default_chat_model
st.session_state["chat_model"] = actual_chat_model
st.sidebar.info(f"Using chat model: {str(actual_chat_model)}")
st.header("DocChatAgent by Langroid", divider="rainbow")

uploaded_file = st.file_uploader("Choose a txt file")
TEMP_DIR = "tempdir"
if uploaded_file is not None:
    if uploaded_file.name != st.session_state["specified_file"]:
        temp_dir = os.makedirs(TEMP_DIR, exist_ok=True)
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["specified_file"] = uploaded_file.name
        st.session_state["file_path"] = temp_path
    else:
        temp_path = st.session_state["file_path"]

temp_path = st.session_state["file_path"]
cfg = configure(temp_path, actual_chat_model)

prompt = st.chat_input("Talk with Document")
if prompt:
    st.write(f"{prompt}")

    # chat using docchatagent
    answer = agent(cfg, prompt)
    st.write(f"{answer}")
