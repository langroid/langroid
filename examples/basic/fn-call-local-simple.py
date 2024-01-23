"""
Function-calling example using a local LLM, with ollama.

"Function-calling" refers to the ability to ability of the LLM to generate
a structured response, typically a JSON object, instead of a plain text response,
which is then interpreted by your code to perform some action.
This is also referred to in various scenarios as "Tools", "Actions" or "Plugins".

# (1) Mac: Install latest ollama, then do this:
# ollama pull mistral:7b-instruct-v0.2-q4_K_M"

# (2) Ensure you've installed the `litellm` extra with Langroid, e.g.
# pip install langroid[litellm], or if you use the `pyproject.toml` in this repo
# you can simply use `poetry install`

# (3) Run like this:

python3 examples/basic/fn-call-local-simple.py

To change the local model, use the optional arg -m <local_model>.
See this [script](https://github.com/langroid/langroid-examples/blob/main/examples/docqa/rag-local-simple.py)
for other ways to specify the local_model.

"""
import os
from typing import List
import fire

from pydantic import BaseModel, Field
import langroid as lr
from langroid.utils.configuration import settings
from langroid.agent.tool_message import ToolMessage
import langroid.language_models as lm
from langroid.agent.chat_document import ChatDocument

# for best results:
DEFAULT_LLM = "litellm/ollama/mixtral:8x7b-instruct-v0.1-q4_K_M"

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
    ) -> str | ChatDocument | None:
        """Fallback method when LLM forgets to generate a tool"""
        if isinstance(msg, ChatDocument) and msg.metadata.sender == "LLM":
            return """
                You must use the "city_tool" to generate city information.
                You either forgot to use it, or you used it with the wrong format.
                Make sure all fields are filled out and pay attention to the 
                required types of the fields.
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


def app(
    m: str = DEFAULT_LLM,
    d: bool = False,
):
    settings.debug = d
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
    # and few-shot examples into the system message
    agent.enable_message(CityTool)

    # (5) Create task and run it to start an interactive loop
    task = lr.Task(agent)
    task.run("Start by asking me for a city name")


if __name__ == "__main__":
    fire.Fire(app)
