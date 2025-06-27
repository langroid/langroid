"""
Barebones example of a single agent using 3 tools.
Similar to 1-agent-3-tools.py, but here the task is set up
with `interactive=False`, meaning user input is awaited only
when user is explicitly addressed using an addressing prefix.
"""

from typing import Any, List, Tuple

import fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.orchestration import ForwardTool
from langroid.utils.configuration import settings
from langroid.utils.constants import AT

DEFAULT_LLM = lm.OpenAIChatModel.GPT4o

# (1) DEFINE THE TOOLS


class UpdateTool(lr.ToolMessage):
    request: str = "update"
    purpose: str = "To update the stored number to the given <number>"
    number: int

    @classmethod
    def examples(cls) -> List["lr.ToolMessage" | Tuple[str, "lr.ToolMessage"]]:
        # Examples that will be compiled into few-shot examples for the LLM.
        # Each example can either be...
        return [
            cls(number=3),  # ... just instances of the tool-class, OR
            (  # ...a tuple of "thought leading to tool", and the tool instance
                "I want to update the stored number to number 4 from the user",
                cls(number=4),
            ),
        ]


class AddTool(lr.ToolMessage):
    request: str = "add"
    purpose: str = "To add the given <number> to the stored number"
    number: int

    @classmethod
    def examples(cls) -> List["lr.ToolMessage" | Tuple[str, "lr.ToolMessage"]]:
        return [
            cls(number=3),
            (
                "I want to add number 10 to the stored number",
                cls(number=10),
            ),
        ]


class ShowTool(lr.ToolMessage):
    request: str = "show"
    purpose: str = "To show the user the stored <number>"

    @classmethod
    def examples(cls) -> List["lr.ToolMessage" | Tuple[str, "lr.ToolMessage"]]:
        return [
            cls(number=3),
            (
                "I want to show the user the stored number 10",
                cls(number=10),
            ),
        ]


# (2) DEFINE THE AGENT, with the tool-handling methods
class NumberAgent(lr.ChatAgent):
    secret: int = 0

    def update(self, msg: UpdateTool) -> str:
        self.secret = msg.number
        return f"Ok I updated the stored number to {msg.number}"

    def add(self, msg: AddTool) -> str:
        self.secret += msg.number
        return f"Added {msg.number} to stored number => {self.secret}"

    def show(self, msg: ShowTool) -> str:
        return f"Inform the user that the SECRET NUMBER is {self.secret}"

    def handle_message_fallback(self, msg: str | lr.ChatDocument) -> Any:
        """
        If we're here it means there was no recognized tool in `msg`.
        So if it was from LLM, use ForwardTool to send to user.
        """
        if isinstance(msg, lr.ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            return ForwardTool(agent="User")


def app(
    m: str = DEFAULT_LLM,  # pass -d <model> to use non-default LLM
    d: bool = False,  # pass -d to enable debug mode (see prompts etc)
    nc: bool = False,  # pass -nc to disable cache-retrieval (i.e. get fresh answers)
):
    settings.debug = d
    settings.cache = not nc
    # create LLM config
    llm_cfg = lm.OpenAIGPTConfig(
        chat_model=m or DEFAULT_LLM,
        chat_context_length=4096,  # set this based on model
        max_output_tokens=100,
        temperature=0.2,
        stream=True,
        timeout=45,
    )

    # (3) CREATE THE AGENT
    agent_config = lr.ChatAgentConfig(
        name="NumberAgent",
        llm=llm_cfg,
        system_message=f"""
        When the user's request matches one of your available tools, use it, 
        otherwise respond directly to the user.
        NOTE: Whenever you want to address the user directly, you MUST
        use "{AT}User", followed by your message. 
        """,
    )

    agent = NumberAgent(agent_config)

    # (4) ENABLE/ATTACH THE TOOLS to the AGENT

    agent.enable_message(UpdateTool)
    agent.enable_message(AddTool)
    agent.enable_message(ShowTool)

    # (5) CREATE AND RUN THE TASK
    task_config = lr.TaskConfig(addressing_prefix=AT)
    task = lr.Task(agent, interactive=False, config=task_config)

    """
    Note: try saying these when it waits for user input:
    
    add 10
    update 50
    add 3
    show 
    """

    task.run()


if __name__ == "__main__":
    fire.Fire(app)
