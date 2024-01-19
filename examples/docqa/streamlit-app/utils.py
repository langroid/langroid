from langroid.agent.special import DocChatAgent, DocChatAgentConfig
from langroid.vector_store.lancedb import LanceDBConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.parsing.parser import ParsingConfig
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
import os
import streamlit as st

OPENAI_KEY = os.environ["OPENAI_API_KEY"]


@st.cache_data
def configure(filename: str, chat_model: str = "") -> DocChatAgentConfig:
    llm_cfg = OpenAIGPTConfig(
        chat_model=chat_model,
    )

    oai_embed_config = OpenAIEmbeddingsConfig(
        model_type="openai",
        model_name="text-embedding-ada-002",
        dims=1536,
    )

    # Configuring DocChatAgent
    cfg = DocChatAgentConfig(
        parsing=ParsingConfig(
            chunk_size=100,
            overlap=20,
            n_similar_docs=4,
        ),
        show_stats=False,
        cross_encoder_reranking_model="",
        llm=llm_cfg,
        vecdb=LanceDBConfig(
            embedding=oai_embed_config,
            collection_name="lease",
            replace_collection=True,
        ),
        doc_paths=[filename],
    )

    return cfg


def agent(cfg, prompt):
    # Creating DocChatAgent
    rag_agent = st.session_state["rag_agent"]
    if (
        rag_agent is None
        or st.session_state["chat_model"] != cfg.llm.chat_model
        or st.session_state["file_path"] != cfg.doc_paths[0]
    ):
        rag_agent = DocChatAgent(cfg)
        st.session_state["rag_agent"] = rag_agent

    response = rag_agent.llm_response(prompt)
    return response.content
