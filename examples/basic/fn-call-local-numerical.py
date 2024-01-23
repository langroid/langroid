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

python3 examples/docqa/fn-call-local-simple.py

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

# (1) Define the desired fn-call as a ToolMessage via Pydantic.

class PolinskyTool(lr.agent.ToolMessage):
    """A fictitious number transformation tool"""
    request: str = "polinsky"
    purpose: str = "To find out the Polinsky transform of a <number>."
    number: int

    def handle(self) -> str:
        """Handle LLM's structured output if it matches Polinsky tool"""
        result = self.number * 3 + 1
        msg = f"""
        SUCCESS! The Polinksy transform of {self.number} is {result}.
        Now ask me for another number.
        """
        return msg


    @staticmethod
    def handle_message_fallback(
        agent: lr.ChatAgent, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """Fallback method when LLM forgets to generate a tool"""
        if isinstance(msg, ChatDocument) and msg.metadata.sender == "LLM":
            return """
                You must use the "polinskty" tool/function to 
                request the Polinsky transform of a number.
                You either forgot to use it, or you used it with the wrong format.
                Make sure all fields are filled out and pay attention to the 
                required types of the fields.
                """

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        # Used to provide few-shot examples in the system prompt
        return [
            cls(
                number=19,
            ),
            cls(
                number=5,
            ),
        ]

def app(
    m: str = DEFAULT_LLM, # model name
    d: bool = False, # debug
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
        You are an expert at calling functions using the specified syntax.
        The user wants to know the Polinsky transform of a number, and you do not
        know how to calculate it. 
        So when the user gives you a number, you must use the `polinsky` function/tool
        to request the Polinsky transform of that number. This will be computed by 
        an assistant, who will return the answer to you. You must then return the answer
        to the user, and ask for another number, and so on.
        
        START BY ASKING ME TO GIVE YOU A NUMBER.
        DO NOT SAY ANYTHING UNTIL YOU GET A NUMBER.
        """,
    )

    agent = lr.ChatAgent(config)

    # (4) Enable the Tool for this agent --> this auto-inserts JSON instructions
    # and few-shot examples into the system message
    agent.enable_message(PolinskyTool)

    # (5) Create task and run it to start an interactive loop
    task = lr.Task(agent)
    task.run("Start by asking me for a number")


if __name__ == "__main__":
    fire.Fire(app)
