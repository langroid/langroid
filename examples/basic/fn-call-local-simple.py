"""
Function-calling example using a local LLM, with ollama.

"Function-calling" refers to the ability to ability of the LLM to generate
a structured response, typically a JSON object, instead of a plain text response,
which is then interpreted by your code to perform some action.
This is also referred to in various scenarios as "Tools", "Actions" or "Plugins".

# (1) Mac: Install latest ollama, then do this:
# ollama pull mistral:7b-instruct-v0.2-q4_K_M

# (2) Ensure you've installed the `litellm` extra with Langroid, e.g.
# pip install langroid[litellm] (or use pip install langroid\[litellm\] if using zsh),
or if you use the `pyproject.toml` in this repo you can simply use `poetry install`

# (3) Run like this:

python3 examples/basic/fn-call-local-simple.py

"""
import os
from typing import List

from pydantic import BaseModel, Field
import langroid as lr
from langroid.agent.tool_message import ToolMessage
import langroid.language_models as lm
from langroid.agent.chat_document import ChatDocument


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create the llm config object.
# Note: if instead of ollama you've spun up your local LLM to listen at
# an OpenAI-Compatible Endpoint like `localhost:8000`, then you can set
# chat_model="local/localhost:8000"; carefully note there's no http in this,
# and if the endpoint is localhost:8000/v1, then you must set
# chat_model="local/localhost:8000/v1"
# Similarly if your endpoint is `http://128.0.4.5:8000/v1`, then you must set
# chat_model="local/128.0.4.5:8000/v1"
llm_cfg = lm.OpenAIGPTConfig(
    chat_model="litellm/ollama/mistral:7b-instruct-v0.2-q4_K_M",
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
    purpose: str = "present <city_info>"
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
    ) -> str | ChatDocument | None:
        """Fallback method when LLM forgets to generate a tool"""
        if isinstance(msg, ChatDocument) and msg.metadata.sender == "LLM":
            return """
            You must use the `city_tool` to generate city information.
            You either forgot to use it, or you used it with the wrong format.
            Make sure all fields are filled out.
            """

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


# (3) Define a ChatAgentConfig and ChatAgent

config = lr.ChatAgentConfig(
    llm=llm_cfg,
    system_message="""
    You are an expert on world city information. 
    The user will give you a city name, and you should use the `city_tool` to
    generate information about the city, and present it to the user.
    Make up the values if you don't know them exactly, but make sure
    the structure is as specified in the `city_tool` JSON definition.
    
    DO NOT SAY ANYTHING ELSE BESIDES PROVIDING THE CITY INFORMATION.
    
    START BY ASKING ME TO GIVE YOU A CITY NAME. 
    DO NOT GENERATE ANYTHING YOU GET A CITY NAME.
    
    Once you've generated the city information using `city_tool`,
    ask for another city name, and so on.
    """,
)

agent = lr.ChatAgent(config)

# (4) Enable the Tool for this agent --> this auto-inserts JSON instructions
# and few-shot examples into the system message
agent.enable_message(CityTool)

# (5) Create task and run it to start an interactive loop
task = lr.Task(agent)
task.run()
