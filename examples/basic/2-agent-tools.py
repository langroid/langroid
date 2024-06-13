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

import langroid as lr

main = lr.ChatAgent(
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
    request = "polinsky"
    purpose = "To compute the polinsky transform of a <number>"
    number: int

    def handle(self):
        p = 3 * self.number + 1
        return f"The Polinsky transform of {self.number} is {p}"


class ChichikovTool(lr.ToolMessage):
    request = "chichikov"
    purpose = "To compute the Chichikov transform of a <number>"
    number: int

    def handle(self):
        n = self.number**2
        return f"The Chichikov transform of {self.number} is {n}"


helper = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="Helper",
        system_message=f"""
        You have a few tools to help answer the user's questions. 
        Decide which tool to use, and send your request using the correct format 
        for the tool.
        """,
    )
)
helper.enable_message(PolinskyTool)
helper.enable_message(ChichikovTool)

main_task = lr.Task(main, interactive=True)
helper_task = lr.Task(helper, interactive=False, done_if_response=[lr.Entity.AGENT])

main_task.add_sub_task(helper_task)

main_task.run()
