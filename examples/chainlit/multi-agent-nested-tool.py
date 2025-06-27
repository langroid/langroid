"""
TODO - this example does not work yet due to breaking changes in Chainlit

2-agent chat, using task.run(), where the sub-task uses a tool to get user input.
This illustrates how a sub-task's steps, including tool-calls, are nested
one level under the parent task's steps.

After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/multi-agent-nested-tool.py
"""

from textwrap import dedent

import chainlit as cl

import langroid as lr
from langroid.agent.callbacks.chainlit import ChainlitTaskCallbacks, add_instructions
from langroid.utils.configuration import settings
from langroid.utils.constants import DONE

settings.cache = False


class ExportTool(lr.ToolMessage):
    request = "main_export"
    purpose = "To request the main export of a given <country>."
    country: str


class StudentChatAgent(lr.ChatAgent):
    def main_export(self, msg: ExportTool) -> str:
        assert (
            self.callbacks.get_user_response is not None
        ), "No get_user_response method"
        assert (
            self.callbacks.show_agent_response is not None
        ), "No show_agent_response method"

        prompt = "Please tell me the main export of " + msg.country
        # create the question for user as an agent response since it
        # will ensure it is shown at right nesting level
        # self.callbacks.show_agent_response(content=prompt)
        user_response = self.callbacks.get_user_response(prompt=prompt)
        res = "the main export is " + user_response
        return res


@cl.on_chat_start
async def on_chat_start():
    await add_instructions(
        title="Two-Agent Demo, where sub-agent uses a Tool/function-call",
        content=dedent(
            """
        **Teacher Agent** delegates to **Student Agent.** 
        - **Teacher** Agent asks a "country export" question to **Student** Agent
        - user (you) hits `c` to continue on to the **Student**
        - **Student** LLM uses `export` tool/fn-call to get answer from user
        - **Student** Agent handler code presents this question to you (user)
        - you answer the question
        - **Student** Agent handler returns your answer
        - **Student** LLM shows the answer
        - user hits `c` to continue on to the **Teacher**
        - **Teacher** Agent gives feedback
        - and so on.
        
        Note how all steps of the (student) sub-task are nested one level below 
        the main (teacher) task.
        """
        ),
    )

    config = lr.ChatAgentConfig()
    teacher_agent = lr.ChatAgent(config)
    teacher_task = lr.Task(
        teacher_agent,
        name="Teacher",
        interactive=True,
        system_message="""
        Ask your student what the main export of a country is, and give feedback. 
        Start with a question!
        """,
    )
    student_agent = StudentChatAgent(config)
    student_agent.enable_message(ExportTool)
    student_task = lr.Task(
        student_agent,
        name="Student",
        interactive=True,
        system_message=f"""
        When you receive a country-export question, 
        use the `main_export` tool to get the answer from the user.
        When you get the answer, say {DONE} and show the answer.
        """,
    )

    teacher_task.add_sub_task(student_task)
    ChainlitTaskCallbacks(teacher_task)
    await teacher_task.run_async()
