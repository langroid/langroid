from typing import Callable, Literal

import chainlit as cl
from _typeshed import Incomplete
from pydantic import BaseSettings

import langroid as lr
import langroid.language_models as lm
from langroid.exceptions import LangroidImportError as LangroidImportError
from langroid.utils.configuration import settings as settings
from langroid.utils.constants import NO_ANSWER as NO_ANSWER

log_level: Incomplete
USER_TIMEOUT: int
SYSTEM: str
LLM: str
AGENT: str
YOU: str
ERROR: str

async def ask_helper(func, **kwargs): ...
async def setup_llm() -> None: ...
async def update_llm(new_settings) -> None: ...
async def make_llm_settings_widgets(
    config: lm.OpenAIGPTConfig | None = None,
) -> None: ...
async def inform_llm_settings() -> None: ...
async def add_instructions(
    title: str = "Instructions",
    content: str = "Enter your question/response in the dialog box below.",
    display: Literal["side", "inline", "page"] = "inline",
) -> None: ...
async def add_image(
    path: str, name: str, display: Literal["side", "inline", "page"] = "inline"
) -> None: ...
async def get_text_files(
    message: cl.Message, extensions: list[str] = [".txt", ".pdf", ".doc", ".docx"]
) -> dict[str, str]: ...
def wrap_text_preserving_structure(text: str, width: int = 90) -> str: ...

class ChainlitCallbackConfig(BaseSettings):
    user_has_agent_name: bool

class ChainlitAgentCallbacks:
    last_step: cl.Step | None
    curr_step: cl.Step | None
    stream: cl.Step | None
    parent_agent: lr.Agent | None
    config: Incomplete
    agent: Incomplete
    def __init__(
        self,
        agent: lr.Agent,
        msg: cl.Message = None,
        config: ChainlitCallbackConfig = ...,
    ) -> None: ...
    def set_parent_agent(self, parent: lr.Agent) -> None: ...
    def get_last_step(self) -> cl.Step | None: ...
    def start_llm_stream(self) -> Callable[[str], None]: ...
    def cancel_llm_stream(self) -> None: ...
    def finish_llm_stream(self, content: str, is_tool: bool = False) -> None: ...
    def show_llm_response(
        self,
        content: str,
        is_tool: bool = False,
        cached: bool = False,
        language: str | None = None,
    ) -> None: ...
    def show_error_message(self, error: str) -> None: ...
    def show_agent_response(self, content: str, language: str = "text") -> None: ...
    def show_start_response(self, entity: str) -> None: ...
    def get_user_response(self, prompt: str) -> str: ...
    def show_user_response(self, message: str) -> None: ...
    def show_first_user_message(self, msg: cl.Message): ...
    async def ask_user_step(
        self, prompt: str, timeout: int = ..., suppress_values: list[str] = ["c"]
    ) -> str: ...

class ChainlitTaskCallbacks(ChainlitAgentCallbacks):
    task: Incomplete
    def __init__(
        self,
        task: lr.Task,
        msg: cl.Message = None,
        config: ChainlitCallbackConfig = ...,
    ) -> None: ...
    last_step: Incomplete
    def show_subtask_response(
        self, task: lr.Task, content: str, is_tool: bool = False
    ) -> None: ...
