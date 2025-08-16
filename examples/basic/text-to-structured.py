"""
Function-calling example using a local LLM, with ollama.

"Function-calling" refers to the ability of the LLM to generate
a structured response, typically a JSON object, instead of a plain text response,
which is then interpreted by your code to perform some action.
This is also referred to in various scenarios as "Tools", "Actions" or "Plugins".
See more here: https://langroid.github.io/langroid/quick-start/chat-agent-tool/

Run like this (to run with llama-3.1-8b-instant via groq):

python3 examples/basic/text-to-structured.py -m groq/llama-3.1-8b-instant

Other models to try it with:
- ollama/qwen2.5-coder
- ollama/qwen2.5


See here for how to set up a Local LLM to work with Langroid:
https://langroid.github.io/langroid/tutorials/local-llm-setup/


"""

import json
import os
from typing import List, Literal

import fire
from rich.prompt import Prompt

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import ResultTool
from pydantic import BaseModel, Field
from langroid.utils.configuration import settings

# for best results:
DEFAULT_LLM = lm.OpenAIChatModel.GPT4o

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# (1) Define the desired structure via Pydantic.
# The "Field" annotations are optional, and are included in the system message
# if provided, and help with generation accuracy.


class Wifi(BaseModel):
    name: str


class HomeSettings(BaseModel):
    App: List[str] = Field(..., description="List of apps found in text")
    wifi: List[Wifi] = Field(..., description="List of wifi networks found in text")
    brightness: Literal["low", "medium", "high"] = Field(
        ..., description="Brightness level found in text"
    )


# (2) Define the Tool class for the LLM to use, to produce the above structure.
class HomeAutomationTool(lr.agent.ToolMessage):
    """Tool to extract Home Automation structure from text"""

    request: str = "home_automation_tool"
    purpose: str = """
    To extract <home_settings> structure from a given text.
    """
    home_settings: HomeSettings = Field(
        ..., description="Home Automation settings from given text"
    )

    def handle(self) -> str:
        """Handle LLM's structured output if it matches HomeAutomationTool structure"""
        print(
            f"""
            SUCCESS! Got Valid Home Automation Settings:
            {json.dumps(self.home_settings.model_dump(), indent=2)}
            """
        )
        return ResultTool(settings=self.home_settings)

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        # Used to provide few-shot examples in the system prompt
        return [
            (
                """
                    I have extracted apps Spotify and Netflix, 
                    wifi HomeWifi, and brightness medium
                    """,
                cls(
                    home_settings=HomeSettings(
                        App=["Spotify", "Netflix"],
                        wifi=[Wifi(name="HomeWifi")],
                        brightness="medium",
                    )
                ),
            )
        ]


def app(
    m: str = DEFAULT_LLM,  # model
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

    tool_name = HomeAutomationTool.default_value("request")
    config = lr.ChatAgentConfig(
        llm=llm_cfg,
        system_message=f"""
        You are an expert in extracting home automation settings from text.
        When user gives a piece of text, use the TOOL `{tool_name}`
        to present the extracted structured information.
        """,
    )

    agent = lr.ChatAgent(config)

    # (4) Enable the Tool for this agent --> this auto-inserts JSON instructions
    # and few-shot examples (specified in the tool defn above) into the system message
    agent.enable_message(HomeAutomationTool)

    # (5) Create task and run it to start an interactive loop
    # Specialize the task to return a ResultTool object
    task = lr.Task(agent, interactive=False)[ResultTool]

    # set up a loop to extract Home Automation settings from text
    while True:
        text = Prompt.ask("[blue]Enter text (or q/x to exit)")
        if not text or text.lower() in ["x", "q"]:
            break
        result = task.run(text)
        assert isinstance(result, ResultTool)
        assert isinstance(result.settings, HomeSettings)


if __name__ == "__main__":
    fire.Fire(app)
