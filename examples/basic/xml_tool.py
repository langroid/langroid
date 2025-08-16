"""
Example of defining a variant of an existing tool, but inheriting from XMLToolMessage,
to have the LLM use XML rather than JSON to generate the tool.

This will not work with built-in functions/tools of OpenAI,
so in the `ChatAgentConfig` , you have to set the following to ensure
that Langroid's built-in XML Tool calls are activated:
- `use_functions_api = False`
- `use_tools = True`

Run like this (--model is optional, defaults to GPT4o):

python3 examples/basic/xml_tool.py --model groq/llama-3.1-8b-instant
"""

import fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.orchestration import SendTool
from langroid.agent.xml_tool_message import XMLToolMessage
from pydantic import Field


class XMLSendTool(SendTool, XMLToolMessage):
    """
    Variant of SendTool, using XML rather than JSON.
    """

    request: str = "xml_send_tool"
    purpose: str = """
        To send <content> to an entity/agent identified in the <to> field.
        """

    content: str = Field(
        ...,
        description="The content to send",
        verbatim=True,  # enforces content enclosed within CDATA block in xml.
    )
    to: str


xml_send_tool_name = XMLSendTool.default_value("request")


def main(model: str = ""):
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
    )
    alice = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="Alice",
            llm=llm_config,
            use_functions_api=False,
            use_tools=True,
            system_message=f"""
            Whatever number you receive, send it to Bob using the  
            `{xml_send_tool_name}` tool.
            """,
        )
    )

    bob = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="Bob",
            llm=llm_config,
            use_functions_api=False,
            use_tools=True,
            system_message=f"""
            Whatever number you receive, add 1 to it and send 
            the result back to Alice
            using the `{xml_send_tool_name}` tool.
            """,
        )
    )

    alice.enable_message(XMLSendTool)
    bob.enable_message(XMLSendTool)

    # specialize alice_task to return an int
    alice_task = lr.Task(alice, interactive=False)[int]
    bob_task = lr.Task(bob, interactive=False)

    alice_task.add_sub_task(bob_task)

    result = alice_task.run("5", turns=6)
    assert result == 7


if __name__ == "__main__":
    fire.Fire(main)
