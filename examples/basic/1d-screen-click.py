"""

A Bit-Shooter Game played on a 1-dimensional binary screen.

Given an LLM Agent access to a 1-dimensional "screen" represented
as a string of bits (0s and 1s), e.g. "101010",
and equip it with a "Click tool" (like a mouse click) that allows it to
click on a bit -- clicking the bit causes it to flip.

The Agent plays a "Bit Shooter" game where the goal is to get rid of all
1s in the "screen".

To use the Click tool, the Agent must specify the position (zero-based)
where it wants to click. This causes the bit to flip.
The LLM is then presented with the new state of the screen,
and the process repeats until all 1s are gone.

Clearly the Agent (LLM) needs to be able to accurately count the bit positions,
to be able to correctly click on the 1s.

Run like this (--model is optional, defaults to GPT4o):

python3 examples/basic/1d-screen-click.py --model litellm/anthropic/claude-3-5-sonnet-20241022

At the beginning you get to specify the initial state of the screen:
- size of the screen (how many bits)
- the (0-based) locations of the 1s (SPACE-separated) in the screen.

E.g. try this:
- size = 50,
- 1-indices: 0 20 30 40

The loop is set to run in interactive mode (to prevent runaway loops),
so you have to keep hitting enter to see the LLM's next move.

The main observation is that when you run it with claude-3.5-sonnet,
the accuracy of the Agent's clicks is far superior to other LLMs like GPT-4o
and even GPT-4.

To try with other LLMs, you can set the --model param to, for example:
- gpt-4 (set OPENAI_API_KEY in your env or .env file)
- gpt-4o (ditto, set OPENAI_API_KEY)
- groq/llama-3.1-70b-versatile (set GROQ_API_KEY in your env or .env file)
- cerebras/llama3.1-70b (set CEREBRAS_API_KEY in your env or .env file)
- ollama/qwen2.5-coder:latest

See here for a full guide on local/open LLM setup with Langroid:
https://langroid.github.io/langroid/tutorials/local-llm-setup/
And here for how to use with other non-OpenAPI LLMs:
https://langroid.github.io/langroid/tutorials/non-openai-llms/
"""

from typing import List, Tuple

import fire
from rich.prompt import Prompt

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.orchestration import AgentDoneTool
from langroid.pydantic_v1 import BaseModel
from langroid.utils.globals import GlobalState


class ScreenState(BaseModel):
    """
    Represents the state of the 1-dimensional binary screen
    """

    screen: str | None = None  # binary string, e.g. "101010"

    def __init__(
        self,
        one_indices: List[int] = [1],
        size: int = 1,
    ):
        super().__init__()
        # Initialize with all zeros
        screen_list = ["0"] * size

        # Set 1s at specified indices
        for idx in one_indices:
            if 0 <= idx < size:
                screen_list[idx] = "1"

        # Join into string
        self.screen = "".join(screen_list)

    @classmethod
    def set_state(
        cls,
        one_indices: List[int],
        size: int,
    ) -> "ScreenState":
        """
        Factory method to create and set initial state.
        """
        initial_state = cls(
            one_indices=one_indices,
            size=size,
        )
        GlobalScreenState.set_values(state=initial_state)

    def flip(self, i: int):
        """
        Flip the i-th bit
        """
        if self.screen is None or i < 0 or i >= len(self.screen):
            return

        screen_list = list(self.screen)
        screen_list[i] = "1" if screen_list[i] == "0" else "0"
        self.screen = "".join(screen_list)


class GlobalScreenState(GlobalState):
    state: ScreenState = ScreenState()


def get_state() -> ScreenState:
    return GlobalScreenState.get_value("state")


class ClickTool(lr.ToolMessage):
    request: str = "click_tool"
    purpose: str = """
        To click at <position> on the 1-dimensional binary screen, 
        which causes the bit at that position to FLIP.
        IMPORTANT: the position numbering starts from 0!!!
    """

    position: int

    @classmethod
    def examples(cls) -> List[lr.ToolMessage | Tuple[str, lr.ToolMessage]]:
        return [
            cls(position=3),
            (
                "I want to click at position 5",
                cls(position=5),
            ),
        ]

    def handle(self) -> str | AgentDoneTool:
        state = get_state()
        state.flip(self.position)
        print("SCREEN STATE = ", state.screen)
        if "1" not in state.screen:
            return AgentDoneTool()
        return state.screen


def main(model: str = ""):
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
    )
    click_tool_name = ClickTool.default_value("request")
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="Clicker",
            llm=llm_config,
            use_functions_api=False,  # suppress OpenAI functions/tools
            use_tools=True,  # enable langroid-native tools: works with any LLM
            show_stats=False,
            system_message=f"""
            You are an expert at COMPUTER USE.
            In this task you only have to be able to understand a 1-dimensional 
            screen presented to you as a string of bits (0s and 1s).
            You will play a 1-dimensional BIT-shooter game!
            
            Your task is to CLICK ON THE LEFTMOST 1 in the bit-string, 
            to flip it to a 0.
            
            Always try to click on the LEFTMOST 1 in the bit-sequence. 
            
            To CLICK on the screen you 
            must use the TOOL `{click_tool_name}` where the  
            `position` field specifies the position (zero-based) to click.
            If you CORRECTLY click on a 1, the bit at that position will be 
            turned to 0.
            But if you click on a 0, it will turn into a 1, 
            taking you further from your goal.
            
            So you MUST ACCURATELY specify the position of the LEFTMOST 1 to click,
            making SURE there is a 1 at that position.
            In other words, it is critical that you are able to ACCURATELY COUNT 
            the bit positions so that you are able to correctly identify the position 
            of the LEFTMOST 1 bit in the "screen" given to you as a string of bits.
            """,
        )
    )

    agent.enable_message(ClickTool)

    task = lr.Task(agent, interactive=True, only_user_quits_root=False)

    # kick it off with initial screen state (set below by user)
    task.run(get_state())


if __name__ == "__main__":
    size = int(Prompt.ask("Size of screen (how many bits)"))
    ones = Prompt.ask("Indices of 1s (SPACE-separated)").split(" ")
    ones = [int(x) for x in ones]
    ScreenState.set_state(ones, size)
    print("SCREEN STATE = ", get_state().screen)
    fire.Fire(main)
