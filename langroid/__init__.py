"""
Main langroid package
"""

import os
import nltk

def download_punkt_venv():
    # Set NLTK_DATA path to the .venv-specific directory
    nltk_data_path = os.path.join(os.path.dirname(__file__), '..', '.venv', 'nltk_data')
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)

    # Update NLTK's default data path
    nltk.data.path.append(nltk_data_path)

    # Check if 'punkt' is available, otherwise download it
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print(f"Downloading 'punkt' tokenizer into {nltk_data_path}...")
        nltk.download('punkt', download_dir=nltk_data_path)

download_punkt_venv()

from . import mytypes
from . import utils

from . import parsing
from . import prompts
from . import cachedb

from . import language_models
from . import embedding_models

from . import vector_store
from . import agent

from .agent.base import (
    Agent,
    AgentConfig,
)

from .agent.batch import (
    run_batch_tasks,
    llm_response_batch,
    agent_response_batch,
)

from .agent.chat_document import (
    StatusCode,
    ChatDocument,
    ChatDocMetaData,
)

from .agent.tool_message import (
    ToolMessage,
)

from .agent.chat_agent import (
    ChatAgent,
    ChatAgentConfig,
)

from .agent.task import Task, TaskConfig


from .mytypes import (
    DocMetaData,
    Document,
    Entity,
)

from .exceptions import InfiniteLoopException
from .exceptions import LangroidImportError

__all__ = [
    "mytypes",
    "exceptions",
    "utils",
    "parsing",
    "prompts",
    "cachedb",
    "language_models",
    "embedding_models",
    "vector_store",
    "agent",
    "Agent",
    "AgentConfig",
    "ChatAgent",
    "ChatAgentConfig",
    "StatusCode",
    "ChatDocument",
    "ChatDocMetaData",
    "Task",
    "TaskConfig",
    "DocMetaData",
    "Document",
    "Entity",
    "ToolMessage",
    "run_batch_tasks",
    "llm_response_batch",
    "agent_response_batch",
    "InfiniteLoopException",
    "LangroidImportError",
]


try:
    from .agent.callbacks.chainlit import (
        ChainlitAgentCallbacks,
        ChainlitTaskCallbacks,
        ChainlitCallbackConfig,
    )

    ChainlitAgentCallbacks
    ChainlitTaskCallbacks
    ChainlitCallbackConfig
    __all__.extend(
        [
            "ChainlitAgentCallbacks",
            "ChainlitTaskCallbacks",
            "ChainlitCallbackConfig",
        ]
    )
except ImportError:
    pass
