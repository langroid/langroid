"""
Function-calling example using a local LLM, with ollama.

"Function-calling" refers to the ability of the LLM to generate
a structured response, typically a JSON object, instead of a plain text response,
which is then interpreted by your code to perform some action.
This is also referred to in various scenarios as "Tools", "Actions" or "Plugins".

This script is designed to have a basic ChatAgent (powered by a LocalLLM)
engage in a multi-round conversation where the user may occasionally
ask for the "Polinsky transform" of a number, which requires the LLM to
use a `Polinsky` tool/function-call. This is a fictitious transform,
that simply does n => 3n + 1.
We intentionally use a fictitious transform rather than something like "square"
or "double" to prevent the LLM from trying to answer the question directly.

The challenging part here is getting the LLM to decide on an appropriate response
to a few different types of user messages:
- user asks a general question -> LLM should answer the question directly
- user asks for the Polinsky transform of a number -> LLM should use the Polinsky tool
- result from applying Polinsky transform -> LLM should present this to the user
- user says there was a format error in using the Polinsky tool -> LLM should try this tool again

Many models quickly get confused in a multi-round conversation like this.
However `dolphin-mixtral` seems to do well here.

Run like this --

python3 examples/basic/fn-call-local-numerical.py -m <local_model_name>

Local model with best results is dolphin-mixtral:
```
ollama run dolphin-mixtral
python3 examples/basic/fn-call-local-numerical.py -m litellm/ollama_chat/dolphin-mixtral:latest
```

See here for how to set up a Local LLM to work with Langroid:
https://langroid.github.io/langroid/tutorials/local-llm-setup/

"""
import os
from typing import List
import fire

import langroid as lr
from langroid.language_models.openai_gpt import OpenAICallParams
from langroid.utils.configuration import settings
from langroid.agent.tool_message import ToolMessage
import langroid.language_models as lm
from langroid.agent.chat_document import ChatDocument

DEFAULT_LLM = lm.OpenAIChatModel.GPT4_TURBO


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# (1) Define the desired fn-call as a ToolMessage via Pydantic.


class PolinskyTool(lr.agent.ToolMessage):
    """A fictitious number transformation tool. We intentionally use
    a fictitious tool rather than something like "square" or "double"
    to prevent the LLM from trying to answer the question directly.
    """

    request: str = "polinsky"
    purpose: str = (
        "To respond to user request for the Polinsky transform of a <number>."
    )
    number: int

    @classmethod
    def json_instructions(cls) -> str:
        inst = super().json_instructions()
        return f"""
        {inst}
        
        ONLY USE THIS TOOL AFTER THE USER ASKS FOR A POLINSKY TRANSFORM.
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


class MyChatAgent(lr.ChatAgent):
    tool_called: bool = False

    def polinsky(self, msg: PolinskyTool) -> str:
        """Handle LLM's structured output if it matches Polinsky tool"""
        self.tool_called = True
        result = msg.number * 3 + 1
        response = f"""
        SUCCESS! The Polinksy transform of {msg.number} is {result}.
        Present this result to me, and ask for another number.
        """
        return response

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """Fallback method when LLM does not generate a tool,
        and agent ends up handling the msg"""
        if isinstance(msg, ChatDocument) and msg.metadata.sender == "LLM":
            if self.tool_called:
                self.tool_called = False
                return "Ask the user what they need help with"
            else:
                return """
                    You must use the "polinsky" tool/function to 
                    request the Polinsky transform of a number.
                    You either forgot to use it, or you used it with the wrong format.
                    Make sure all fields are filled out and pay attention to the 
                    required types of the fields.
                    """


def app(
    m: str = DEFAULT_LLM,  # model name
    d: bool = False,  # debug
    nc: bool = False,  # no cache
):
    settings.debug = d
    settings.cache = not nc
    # create LLM config
    llm_cfg = lm.OpenAIGPTConfig(
        chat_model=m or DEFAULT_LLM,
        chat_context_length=16_000,  # for dolphin-mixtral
        max_output_tokens=100,
        params=OpenAICallParams(
            presence_penalty=0.8,
            frequency_penalty=0.8,
        ),
        temperature=0,
        stream=True,
        timeout=100,
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
        You are an expert at deciding when to call 
        specified functions with the right syntax.
        You are very very CONCISE in your responses.
        
        Here is how you must respond to my messages:
        
        1. When I ask a general question, simply respond as you see fit.
            Example: 
                ME(User): "What is 3 + 4?"
                YOU(Assistant): "the answer is 7"
                
        2. When I ask to find the Polinksy transform of a number, 
            you  must use the `polinsky` function/tool
            to request the Polinsky transform of that number.
            Example:
                ME(User): "What is the Polinsky transform of 5?"
                YOU(Assistant): <polinsky tool request in JSON format>
                 
        3. When you receive a SUCCESS message with the result from the `polinsky` 
            tool, you must present the result to me in a nice way (CONCISELY), 
            and ask: 'What else can I help with?'
            Example:
                ME(User): "SUCCESS! The Polinksy transform of 5 is 16"
                YOU(Assistant): "The polinsky transform of 5 is 16. What else can I help with?"
                ME(User): "The answer is 16. What is the Polinsky transform of 19?"
                YOU(Assistant): <polinsky tool request in JSON format>
        4. When I say there was an error in using the `polinsky` function/tool,
           you must try the function/tool again with the same number.
              Example:
               ME(User): "There was an error in your use of the polinsky tool:..."
               YOU(Assistant): <polinsky tool request in JSON format>
        """,
    )

    agent = MyChatAgent(config)

    # (4) Enable the Tool for this agent --> this auto-inserts JSON instructions
    # and few-shot examples into the system message
    agent.enable_message(PolinskyTool)

    # (5) Create task and run it to start an interactive loop
    task = lr.Task(agent)
    task.run("Can you help me with some questions?")


if __name__ == "__main__":
    fire.Fire(app)
