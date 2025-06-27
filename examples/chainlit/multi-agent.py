"""
2-agent chat, using task.run(), where:
- Teacher Agent asks a question
- Student Agent answers the question
- Teacher Agent gives feedback
- ...


After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/multi-agent.py
"""

import os
from textwrap import dedent

import chainlit as cl

import langroid as lr
from langroid.agent.callbacks.chainlit import ChainlitTaskCallbacks, add_instructions
from langroid.utils.configuration import settings


@cl.on_chat_start
async def on_chat_start(
    debug: bool = os.getenv("DEBUG", False),
    no_cache: bool = os.getenv("NOCACHE", False),
):
    settings.debug = debug
    settings.cache = not no_cache

    await add_instructions(
        title="Two-Agent Demo",
        content=dedent(
            """
        **Teacher Agent** delegates to **Student Agent.**
        - **Teacher** Agent asks a numerical question to **Student** Agent
        - **Student** Agent answers the question
        - **Teacher** Agent gives feedback        
        - and so on until 10 turns are done.
        
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
        interactive=False,
        system_message="""
        Ask your student concise numerical questions, and give feedback. 
        Start with a question!
        """,
    )
    student_agent = lr.ChatAgent(config)
    student_task = lr.Task(
        student_agent,
        name="Student",
        interactive=False,
        system_message="""Concisely answer your teacher's numerical questions""",
        single_round=True,
    )

    teacher_task.add_sub_task(student_task)
    ChainlitTaskCallbacks(teacher_task)
    await teacher_task.run_async(turns=10)
