"""
Example of defining a variant of an existing tool, but inheriting from XMLToolMessage,
to have the LLM use XML rather than JSON to generate the tool.

This will not work with built-in functions/tools of OpenAI,
so in the `ChatAgentConfig` , you have to set the following to ensure
that Langroid's built-in XML Tool calls are activated:
- `use_functions_api = False`
- `use_tools = True`
"""

import langroid as lr
from langroid.pydantic_v1 import Field
from langroid.agent.tools.orchestration import SendTool, DoneTool
from langroid.agent.xml_tool_message import XMLToolMessage


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
done_tool_name = DoneTool.default_value("request")

alice = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="Alice",
        use_functions_api=False,
        use_tools=True,
        system_message=f"""
        Whatever number you receive, send it to Bob using the 
        `{xml_send_tool_name}` tool. 
        When you receive a number back from Bob, 
        indicate you're done using the `{done_tool_name}` tool,
        with the `content` field containing the number you got from Bob.
        """,
    )
)

bob = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="Bob",
        use_functions_api=False,
        use_tools=True,
        system_message=f"""
        Whatever number you receive, add 1 to it and send it back to Alice
        using the `{xml_send_tool_name}` tool.
        """,
    )
)

alice.enable_message([DoneTool, XMLSendTool])
bob.enable_message(XMLSendTool)

# specialize alice_task to return an int
alice_task = lr.Task(alice, interactive=False)[int]
bob_task = lr.Task(bob, interactive=False)

alice_task.add_sub_task(bob_task)

result = alice_task.run("5")
assert result == 6
