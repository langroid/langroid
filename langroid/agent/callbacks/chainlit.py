"""
Callbacks for Chainlit integration.
"""

import json
import logging
import textwrap
from typing import Any, Callable, Dict, List, Literal, Optional, no_type_check

from langroid.exceptions import LangroidImportError
from langroid.pydantic_v1 import BaseSettings

try:
    import chainlit as cl
except ImportError:
    raise LangroidImportError("chainlit", "chainlit")

from chainlit import run_sync
from chainlit.config import config
from chainlit.logger import logger

import langroid as lr
import langroid.language_models as lm
from langroid.utils.configuration import settings
from langroid.utils.constants import NO_ANSWER

# Attempt to reconfigure the root logger to your desired settings
log_level = logging.INFO if settings.debug else logging.WARNING
logger.setLevel(log_level)
logging.basicConfig(level=log_level)

logging.getLogger().setLevel(log_level)

USER_TIMEOUT = 60_000
SYSTEM = "System 🖥️"
LLM = "LLM 🧠"
AGENT = "Agent <>"
YOU = "You 😃"
ERROR = "Error 🚫"


@no_type_check
async def ask_helper(func, **kwargs):
    res = await func(**kwargs).send()
    while not res:
        res = await func(**kwargs).send()
    return res


@no_type_check
async def setup_llm() -> None:
    """From the session `llm_settings`, create new LLMConfig and LLM objects,
    save them in session state."""
    llm_settings = cl.user_session.get("llm_settings", {})
    model = llm_settings.get("chat_model")
    context_length = llm_settings.get("context_length", 16_000)
    temperature = llm_settings.get("temperature", 0.2)
    timeout = llm_settings.get("timeout", 90)
    logger.info(f"Using model: {model}")
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        # or, other possibilities for example:
        # "litellm/ollama_chat/mistral"
        # "litellm/ollama_chat/mistral:7b-instruct-v0.2-q8_0"
        # "litellm/ollama/llama2"
        # "local/localhost:8000/v1"
        # "local/localhost:8000"
        chat_context_length=context_length,  # adjust based on model
        temperature=temperature,
        timeout=timeout,
    )
    llm = lm.OpenAIGPT(llm_config)
    cl.user_session.set("llm_config", llm_config)
    cl.user_session.set("llm", llm)


@no_type_check
async def update_llm(new_settings: Dict[str, Any]) -> None:
    """Update LLMConfig and LLM from settings, and save in session state."""
    cl.user_session.set("llm_settings", new_settings)
    await inform_llm_settings()
    await setup_llm()


async def make_llm_settings_widgets(
    config: lm.OpenAIGPTConfig | None = None,
) -> None:
    config = config or lm.OpenAIGPTConfig()
    await cl.ChatSettings(
        [
            cl.input_widget.TextInput(
                id="chat_model",
                label="Model Name (Default GPT-4o)",
                initial="",
                placeholder="E.g. ollama/mistral or " "local/localhost:8000/v1",
            ),
            cl.input_widget.NumberInput(
                id="context_length",
                label="Chat Context Length",
                initial=config.chat_context_length,
                placeholder="E.g. 16000",
            ),
            cl.input_widget.Slider(
                id="temperature",
                label="LLM temperature",
                min=0.0,
                max=1.0,
                step=0.1,
                initial=config.temperature,
                tooltip="Adjust based on model",
            ),
            cl.input_widget.Slider(
                id="timeout",
                label="Timeout (seconds)",
                min=10,
                max=200,
                step=10,
                initial=config.timeout,
                tooltip="Timeout for LLM response, in seconds.",
            ),
        ]
    ).send()  # type: ignore


@no_type_check
async def inform_llm_settings() -> None:
    llm_settings: Dict[str, Any] = cl.user_session.get("llm_settings", {})
    settings_dict = dict(
        model=llm_settings.get("chat_model"),
        context_length=llm_settings.get("context_length"),
        temperature=llm_settings.get("temperature"),
        timeout=llm_settings.get("timeout"),
    )
    await cl.Message(
        author=SYSTEM,
        content="LLM settings updated",
        elements=[
            cl.Text(
                name="settings",
                display="side",
                content=json.dumps(settings_dict, indent=4),
                language="json",
            )
        ],
    ).send()


async def add_instructions(
    title: str = "Instructions",
    content: str = "Enter your question/response in the dialog box below.",
    display: Literal["side", "inline", "page"] = "inline",
) -> None:
    await cl.Message(
        author="",
        content=title if display == "side" else "",
        elements=[
            cl.Text(
                name=title,
                content=content,
                display=display,
            )
        ],
    ).send()


async def add_image(
    path: str,
    name: str,
    display: Literal["side", "inline", "page"] = "inline",
) -> None:
    await cl.Message(
        author="",
        content=name if display == "side" else "",
        elements=[
            cl.Image(
                name=name,
                path=path,
                display=display,
            )
        ],
    ).send()


async def get_text_files(
    message: cl.Message,
    extensions: List[str] = [".txt", ".pdf", ".doc", ".docx"],
) -> Dict[str, str]:
    """Get dict (file_name -> file_path) from files uploaded in chat msg"""

    files = [file for file in message.elements if file.path.endswith(tuple(extensions))]
    return {file.name: file.path for file in files}


def wrap_text_preserving_structure(text: str, width: int = 90) -> str:
    """Wrap text preserving paragraph breaks. Typically used to
    format an agent_response output, which may have long lines
    with no newlines or paragraph breaks."""

    paragraphs = text.split("\n\n")  # Split the text into paragraphs
    wrapped_text = []

    for para in paragraphs:
        if para.strip():  # If the paragraph is not just whitespace
            # Wrap this paragraph and add it to the result
            wrapped_paragraph = textwrap.fill(para, width=width)
            wrapped_text.append(wrapped_paragraph)
        else:
            # Preserve paragraph breaks
            wrapped_text.append("")

    return "\n\n".join(wrapped_text)


class ChainlitCallbackConfig(BaseSettings):
    user_has_agent_name: bool = True  # show agent name in front of "YOU" ?
    show_subtask_response: bool = True  # show sub-task response as a step?


class ChainlitAgentCallbacks:
    """Inject Chainlit callbacks into a Langroid Agent"""

    last_step: Optional[cl.Step] = None  # used to display sub-steps under this
    curr_step: Optional[cl.Step] = None  # used to update an initiated step
    stream: Optional[cl.Step] = None  # pushed into openai_gpt.py to stream tokens
    parent_agent: Optional[lr.Agent] = None  # used to get parent id, for step nesting

    def __init__(
        self,
        agent: lr.Agent,
        msg: cl.Message = None,
        config: ChainlitCallbackConfig = ChainlitCallbackConfig(),
    ):
        """Add callbacks to the agent, and save the initial message,
        so we can alter the display of the first user message.
        """
        agent.callbacks.start_llm_stream = self.start_llm_stream
        agent.callbacks.cancel_llm_stream = self.cancel_llm_stream
        agent.callbacks.finish_llm_stream = self.finish_llm_stream
        agent.callbacks.show_llm_response = self.show_llm_response
        agent.callbacks.show_agent_response = self.show_agent_response
        agent.callbacks.get_user_response = self.get_user_response
        agent.callbacks.get_last_step = self.get_last_step
        agent.callbacks.set_parent_agent = self.set_parent_agent
        agent.callbacks.show_error_message = self.show_error_message
        agent.callbacks.show_start_response = self.show_start_response
        self.config = config

        self.agent: lr.Agent = agent
        if msg is not None:
            self.show_first_user_message(msg)

    def _get_parent_id(self) -> str | None:
        """Get step id under which we need to nest the current step:
        This should be the parent Agent's last_step.
        """
        if self.parent_agent is None:
            logger.info(f"No parent agent found for {self.agent.config.name}")
            return None
        logger.info(
            f"Parent agent found for {self.agent.config.name} = "
            f"{self.parent_agent.config.name}"
        )
        last_step = self.parent_agent.callbacks.get_last_step()
        if last_step is None:
            logger.info(f"No last step found for {self.parent_agent.config.name}")
            return None
        logger.info(
            f"Last step found for {self.parent_agent.config.name} = {last_step.id}"
        )
        return last_step.id  # type: ignore

    def set_parent_agent(self, parent: lr.Agent) -> None:
        self.parent_agent = parent

    def get_last_step(self) -> Optional[cl.Step]:
        return self.last_step

    def start_llm_stream(self) -> Callable[[str], None]:
        """Returns a streaming fn that can be passed to the LLM class"""
        self.stream = cl.Step(
            id=self.curr_step.id if self.curr_step is not None else None,
            name=self._entity_name("llm"),
            type="llm",
            parent_id=self._get_parent_id(),
        )
        self.last_step = self.stream
        self.curr_step = None
        logger.info(
            f"""
            Starting LLM stream for {self.agent.config.name}
            id = {self.stream.id} 
            under parent {self._get_parent_id()}
        """
        )
        run_sync(self.stream.send())  # type: ignore

        def stream_token(t: str) -> None:
            if self.stream is None:
                raise ValueError("Stream not initialized")
            run_sync(self.stream.stream_token(t))

        return stream_token

    def cancel_llm_stream(self) -> None:
        """Called when cached response found."""
        self.last_step = None
        if self.stream is not None:
            run_sync(self.stream.remove())  # type: ignore

    def finish_llm_stream(self, content: str, is_tool: bool = False) -> None:
        """Update the stream, and display entire response in the right language."""
        if self.agent.llm is None or self.stream is None:
            raise ValueError("LLM or stream not initialized")
        if content == "":
            run_sync(self.stream.remove())  # type: ignore
        else:
            run_sync(self.stream.update())  # type: ignore
        stream_id = self.stream.id if content else None
        step = cl.Step(
            id=stream_id,
            name=self._entity_name("llm", tool=is_tool),
            type="llm",
            parent_id=self._get_parent_id(),
            language="json" if is_tool else None,
        )
        step.output = textwrap.dedent(content) or NO_ANSWER
        logger.info(
            f"""
            Finish STREAM LLM response for {self.agent.config.name}
            id = {step.id} 
            under parent {self._get_parent_id()}
            """
        )
        run_sync(step.update())  # type: ignore

    def show_llm_response(
        self,
        content: str,
        is_tool: bool = False,
        cached: bool = False,
        language: str | None = None,
    ) -> None:
        """Show non-streaming LLM response."""
        step = cl.Step(
            id=self.curr_step.id if self.curr_step is not None else None,
            name=self._entity_name("llm", tool=is_tool, cached=cached),
            type="llm",
            parent_id=self._get_parent_id(),
            language=language or ("json" if is_tool else None),
        )
        self.last_step = step
        self.curr_step = None
        step.output = textwrap.dedent(content) or NO_ANSWER
        logger.info(
            f"""
            Showing NON-STREAM LLM response for {self.agent.config.name}
            id = {step.id} 
            under parent {self._get_parent_id()}
            """
        )
        run_sync(step.send())  # type: ignore

    def show_error_message(self, error: str) -> None:
        """Show error message as a step."""
        step = cl.Step(
            name=self.agent.config.name + f"({ERROR})",
            type="run",
            parent_id=self._get_parent_id(),
            language="text",
        )
        self.last_step = step
        step.output = error
        run_sync(step.send())

    def show_agent_response(self, content: str, language="text") -> None:
        """Show message from agent (typically tool handler).
        Agent response can be considered as a "step"
        between LLM response and user response
        """
        step = cl.Step(
            id=self.curr_step.id if self.curr_step is not None else None,
            name=self._entity_name("agent"),
            type="tool",
            parent_id=self._get_parent_id(),
            language=language,
        )
        if language == "text":
            content = wrap_text_preserving_structure(content, width=90)
        self.last_step = step
        self.curr_step = None
        step.output = content
        logger.info(
            f"""
            Showing AGENT response for {self.agent.config.name}
            id = {step.id} 
            under parent {self._get_parent_id()}
            """
        )
        run_sync(step.send())  # type: ignore

    def show_start_response(self, entity: str) -> None:
        """When there's a potentially long-running process, start a step,
        so that the UI displays a spinner while the process is running."""
        if self.curr_step is not None:
            run_sync(self.curr_step.remove())  # type: ignore
        step = cl.Step(
            name=self._entity_name(entity),
            type="run",
            parent_id=self._get_parent_id(),
            language="text",
        )
        step.output = ""
        self.last_step = step
        self.curr_step = step
        logger.info(
            f"""
            Showing START response for {self.agent.config.name} ({entity})
            id = {step.id} 
            under parent {self._get_parent_id()}
            """
        )
        run_sync(step.send())  # type: ignore

    def _entity_name(
        self, entity: str, tool: bool = False, cached: bool = False
    ) -> str:
        """Construct name of entity to display as Author of a step"""
        tool_indicator = " =>  🛠️" if tool else ""
        cached = "(cached)" if cached else ""
        match entity:
            case "llm":
                model = self.agent.config.llm.chat_model
                return (
                    self.agent.config.name + f"({LLM} {model} {tool_indicator}){cached}"
                )
            case "agent":
                return self.agent.config.name + f"({AGENT})"
            case "user":
                if self.config.user_has_agent_name:
                    return self.agent.config.name + f"({YOU})"
                else:
                    return YOU
            case _:
                return self.agent.config.name + f"({entity})"

    def _get_user_response_buttons(self, prompt: str) -> str:
        """Not used. Save for future reference"""
        res = run_sync(
            ask_helper(
                cl.AskActionMessage,
                content="Continue, exit or say something?",
                actions=[
                    cl.Action(
                        name="continue",
                        value="continue",
                        label="✅ Continue",
                    ),
                    cl.Action(
                        name="feedback",
                        value="feedback",
                        label="💬 Say something",
                    ),
                    cl.Action(name="exit", value="exit", label="🔚 Exit Conversation"),
                ],
            )
        )
        if res.get("value") == "continue":
            return ""
        if res.get("value") == "exit":
            return "x"
        if res.get("value") == "feedback":
            return self.get_user_response(prompt)
        return ""  # process the "feedback" case here

    def get_user_response(self, prompt: str) -> str:
        """Ask for user response, wait for it, and return it,
        as a cl.Step rather than as a cl.Message so we can nest it
        under the parent step.
        """
        return run_sync(self.ask_user_step(prompt=prompt, suppress_values=["c"]))

    def show_user_response(self, message: str) -> None:
        """Show user response as a step."""
        step = cl.Step(
            id=cl.context.current_step.id,
            name=self._entity_name("user"),
            type="run",
            parent_id=self._get_parent_id(),
        )
        step.output = message
        logger.info(
            f"""
            Showing USER response for {self.agent.config.name}
            id = {step.id} 
            under parent {self._get_parent_id()}
            """
        )
        run_sync(step.send())

    def show_first_user_message(self, msg: cl.Message):
        """Show first user message as a step."""
        step = cl.Step(
            id=msg.id,
            name=self._entity_name("user"),
            type="run",
            parent_id=self._get_parent_id(),
        )
        self.last_step = step
        step.output = msg.content
        run_sync(step.update())

    async def ask_user_step(
        self,
        prompt: str,
        timeout: int = USER_TIMEOUT,
        suppress_values: List[str] = ["c"],
    ) -> str:
        """
        Ask user for input, as a step nested under parent_id.
        Rather than rely entirely on AskUserMessage (which doesn't let us
        nest the question + answer under a step), we instead create fake
        steps for the question and answer, and only rely on AskUserMessage
        with an empty prompt to await user response.

        Args:
            prompt (str): Prompt to display to user
            timeout (int): Timeout in seconds
            suppress_values (List[str]): List of values to suppress from display
                (e.g. "c" for continue)

        Returns:
            str: User response
        """

        # save hide_cot status to restore later
        # (We should probably use a ctx mgr for this)
        hide_cot = config.ui.hide_cot

        # force hide_cot to False so that the user question + response is visible
        config.ui.hide_cot = False

        if prompt != "":
            # Create a question step to ask user
            question_step = cl.Step(
                name=f"{self.agent.config.name} (AskUser ❓)",
                type="run",
                parent_id=self._get_parent_id(),
            )
            question_step.output = prompt
            await question_step.send()  # type: ignore

        # Use AskUserMessage to await user response,
        # but with an empty prompt so the question is not visible,
        # but still pauses for user input in the input box.
        res = await cl.AskUserMessage(
            content="",
            timeout=timeout,
        ).send()

        if res is None:
            run_sync(
                cl.Message(
                    content=f"Timed out after {USER_TIMEOUT} seconds. Exiting."
                ).send()
            )
            return "x"

        # The above will try to display user response in res
        # but we create fake step with same id as res and
        # erase it using empty output so it's not displayed
        step = cl.Step(
            id=res["id"],
            name="TempUserResponse",
            type="run",
            parent_id=self._get_parent_id(),
        )
        step.output = ""
        await step.update()  # type: ignore

        # Finally, reproduce the user response at right nesting level
        if res["output"] in suppress_values:
            config.ui.hide_cot = hide_cot  # restore original value
            return ""

        step = cl.Step(
            name=self._entity_name(entity="user"),
            type="run",
            parent_id=self._get_parent_id(),
        )
        step.output = res["output"]
        await step.send()  # type: ignore
        config.ui.hide_cot = hide_cot  # restore original value
        return res["output"]


class ChainlitTaskCallbacks(ChainlitAgentCallbacks):
    """
    Recursively inject ChainlitAgentCallbacks into a Langroid Task's agent and
    agents of sub-tasks.
    """

    def __init__(
        self,
        task: lr.Task,
        msg: cl.Message = None,
        config: ChainlitCallbackConfig = ChainlitCallbackConfig(),
    ):
        """Inject callbacks recursively, ensuring msg is passed to the
        top-level agent"""

        super().__init__(task.agent, msg, config)
        self._inject_callbacks(task)
        self.task = task
        if config.show_subtask_response:
            self.task.callbacks.show_subtask_response = self.show_subtask_response

    @classmethod
    def _inject_callbacks(
        cls, task: lr.Task, config: ChainlitCallbackConfig = ChainlitCallbackConfig()
    ) -> None:
        # recursively apply ChainlitAgentCallbacks to agents of sub-tasks
        for t in task.sub_tasks:
            cls(t, config=config)
            # ChainlitTaskCallbacks(t, config=config)

    def show_subtask_response(
        self, task: lr.Task, content: str, is_tool: bool = False
    ) -> None:
        """Show sub-task response as a step, nested at the right level."""

        # The step should nest under the calling agent's last step
        step = cl.Step(
            name=self.task.agent.config.name + f"( ⏎ From {task.agent.config.name})",
            type="run",
            parent_id=self._get_parent_id(),
            language="json" if is_tool else None,
        )
        step.output = content or NO_ANSWER
        self.last_step = step
        run_sync(step.send())
