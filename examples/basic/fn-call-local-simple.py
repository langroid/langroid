"""
Function-calling example using a local LLM, with ollama.

"Function-calling" refers to the ability of the LLM to generate
a structured response, typically a JSON object, instead of a plain text response,
which is then interpreted by your code to perform some action.
This is also referred to in various scenarios as "Tools", "Actions" or "Plugins".
See more here: https://langroid.github.io/langroid/quick-start/chat-agent-tool/

Run like this (to run with llama-3.1-8b-instant via groq):

python3 examples/basic/fn-call-local-simple.py -m groq/llama-3.1-8b-instant

See here for how to set up a Local LLM to work with Langroid:
https://langroid.github.io/langroid/tutorials/local-llm-setup/


"""

import os
from typing import List
import fire

from langroid.pydantic_v1 import BaseModel, Field
import langroid as lr
from langroid.utils.configuration import settings
from langroid.agent.tool_message import ToolMessage
import langroid.language_models as lm
from langroid.agent.tools.orchestration import ForwardTool
from langroid.agent.chat_document import ChatDocument

# for best results:
DEFAULT_LLM = lm.OpenAIChatModel.GPT4o

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# (1) Define the desired structure via Pydantic.
# Here we define a nested structure for City information.
# The "Field" annotations are optional, and are included in the system message
# if provided, and help with generation accuracy.


class CityData(BaseModel):
    population: int = Field(..., description="population of city")
    country: str = Field(..., description="country of city")


class City(BaseModel):
    name: str = Field(..., description="name of city")
    details: CityData = Field(..., description="details of city")


# (2) Define the Tool class for the LLM to use, to produce the above structure.
class CityTool(lr.agent.ToolMessage):
    """Present information about a city"""

    request: str = "city_tool"
    purpose: str = """
    To present <city_info> AFTER user gives a city name,
    with all fields of the appropriate type filled out;
    DO NOT USE THIS TOOL TO ASK FOR A CITY NAME.
    SIMPLY ASK IN NATURAL LANGUAGE.
    """
    city_info: City = Field(..., description="information about a city")

    def handle(self) -> str:
        """Handle LLM's structured output if it matches City structure"""
        print("SUCCESS! Got Valid City Info")
        return """
            Thanks! ask me for another city name, do not say anything else
            until you get a city name.
            """

    @staticmethod
    def handle_message_fallback(
        agent: lr.ChatAgent, msg: str | ChatDocument
    ) -> ForwardTool:
        """
        We end up here when there was no recognized tool msg from the LLM;
        In this case forward the message to the user using ForwardTool.
        """
        if isinstance(msg, ChatDocument) and msg.metadata.sender == "LLM":
            return ForwardTool(agent="User")

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        # Used to provide few-shot examples in the system prompt
        return [
            cls(
                city_info=City(
                    name="San Francisco",
                    details=CityData(
                        population=800_000,
                        country="USA",
                    ),
                )
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

    # Recommended: First test if basic chat works with this llm setup as below:
    # Once this works, then you can try the rest of the example.
    #
    # agent = lr.ChatAgent(
    #     lr.ChatAgentConfig(
    #         llm=llm_cfg,
    #     )
    # )
    #
    # agent.llm_response("What is 3 + 4?")
    #
    # task = lr.Task(agent)
    # verify you can interact with this in a chat loop on cmd line:
    # task.run("Concisely answer some questions")

    # Define a ChatAgentConfig and ChatAgent

    config = lr.ChatAgentConfig(
        llm=llm_cfg,
        system_message="""
        You are an expert on world city information. 
        We will play this game, taking turns:
        YOU: ask me to give you a city name.
        I: will give you a city name.
        YOU: use the "city_tool" to generate information about the city, and present it to me.
            Make up the values if you don't know them exactly, but make sure
        I: will confirm whether you provided the info in a valid form,
            and if not I will ask you to try again.
        YOU: wait for my confirmation, and then ask for another city name, and so on.
        
        
        START BY ASKING ME TO GIVE YOU A CITY NAME. 
        DO NOT SAY ANYTHING UNTIL YOU GET A CITY NAME.

        """,
    )

    agent = lr.ChatAgent(config)

    # (4) Enable the Tool for this agent --> this auto-inserts JSON instructions
    # and few-shot examples (specified in the tool defn above) into the system message
    agent.enable_message(CityTool)

    # (5) Create task and run it to start an interactive loop
    task = lr.Task(agent, interactive=False)
    task.run("Start by asking me for a city name")


if __name__ == "__main__":
    fire.Fire(app)
