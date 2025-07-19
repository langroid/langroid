"""
2 Agent setup where Main agent asks a question, Helper has a few tools to help answer,
and for any question, Helper finishes after first use of any tool.

Run like this:

python3 examples/basic/2-agent-tools.py

When it waits for user input, try asking things like:

- capital of uganda?
    => Main answers
- polinsky of 4?
    => Main says do not know, handled by helper, who returns answer
- chichikov of 5?
    => Main says do not know, handled by helper, who returns answer
"""

from typing import Any

import langroid as lr
from langroid.agent.tools.orchestration import AgentDoneTool, ForwardTool


class MainChatAgent(lr.ChatAgent):
    def handle_message_fallback(self, msg: str | lr.ChatDocument) -> Any:
        """
        We'd be here if there were no recognized tools in the incoming msg.
        If this was from LLM, forward to user.
        """
        if isinstance(msg, lr.ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            return ForwardTool(agent="User")


main = MainChatAgent(
    lr.ChatAgentConfig(
        name="Main",
        system_message=f"""
        Help the user with their questions. When you don't know the answer, 
        simply say {lr.utils.constants.NO_ANSWER} and nothing else.
        Your Helper will attempt to handle the question, and send you back their
        answer, and you can present it to the user.   
        
        At the BEGINNING, ask the user what they need help with.
        """,
    )
)


class PolinskyTool(lr.ToolMessage):
    request: str = "polinsky"
    purpose: str = "To compute the polinsky transform of a <number>"
    number: int

    def handle(self) -> AgentDoneTool:
        p = 3 * self.number + 1
        return AgentDoneTool(content=f"The Polinsky transform of {self.number} is {p}")


class ChichikovTool(lr.ToolMessage):
    request: str = "chichikov"
    purpose: str = "To compute the Chichikov transform of a <number>"
    number: int

    def handle(self) -> AgentDoneTool:
        n = self.number**2
        return AgentDoneTool(content=f"The Chichikov transform of {self.number} is {n}")


helper = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="Helper",
        system_message="""
        You have a few tools to help answer the user's questions. 
        Decide which tool to use, and send your request using the correct format 
        for the tool.
        """,
    )
)
helper.enable_message(PolinskyTool)
helper.enable_message(ChichikovTool)

main_task = lr.Task(main, interactive=False)
helper_task = lr.Task(helper, interactive=False)

main_task.add_sub_task(helper_task)

main_task.run()
