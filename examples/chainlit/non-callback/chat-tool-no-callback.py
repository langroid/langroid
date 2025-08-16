"""
Basic single-agent chat example, using a Tool, without streaming.
DEPCRECATED: Script kept only for reference.
The better way is shown in chat-task-tool, which uses ChainlitTaskCallbacks.

- User enters a country
- LLM responds with a tool/function-call showing {country=country, capital=...}
- Agent handler recognizes this tool and returns plain text version of the tool result.

After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/chat-tool-no-callback.py
"""

import chainlit as cl

import langroid as lr


class CapitalTool(lr.ToolMessage):
    request: str = "capital"
    purpose: str = "To present the capital of given <country>."
    country: str
    capital: str

    def handle(self) -> str:
        return f"""
        Success! LLM responded with a tool/function-call, with result:
        Capital of {self.country} is {self.capital}.
        """


@cl.step
async def llm_tool_call(msg: str) -> lr.ChatDocument:
    agent: lr.ChatAgent = cl.user_session.get("agent")
    response = await agent.llm_response_async(msg)
    return response


@cl.on_chat_start
async def on_chat_start():
    sys_msg = """
        You are an expert in country capitals.
        When user gives a country name, you should respond 
        with the capital of that country, using the `capital` tool/function-call.
    """
    config = lr.ChatAgentConfig(
        system_message=sys_msg,
    )
    agent = lr.ChatAgent(config)
    agent.enable_message(CapitalTool)
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent: lr.ChatAgent = cl.user_session.get("agent")
    msg = cl.Message(content="")
    # expecting a tool here
    tool = await llm_tool_call(message.content)
    tool_result = await agent.agent_response_async(tool)
    msg.content = tool_result.content
    await msg.send()
