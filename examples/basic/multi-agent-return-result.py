"""
3-agent system where Main task has subtasks that are able to directly return final
task result, "short-circuiting" the flow.

main_task has sub-tasks even_task and odd_task.

- main_task receives a number, simply passes it on.
- even_task can only handle even number N, returns N/2 as final result, 
    else passes it on.
- odd_task can only handle odd number N, returns 3N+1 as final result, 
    else passes it on.
"""

import langroid as lr
from langroid.agent.tools.orchestration import FinalResultTool

main_agent = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="Main",
        system_message="Whatever number you receive, simply repeat it",
    )
)


class MyFinalResultTool(FinalResultTool):
    request: str = "my_final_result_tool"
    purpose: str = "To present the final result of the exercise"
    _allow_llm_use: bool = True

    answer: int  # could of course be str if answer is text


my_final_result_tool = MyFinalResultTool.default_value("request")

even_agent = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="Even",
        system_message=f"""
        - If you receive an even number, return half of it using the 
          TOOL `{my_final_result_tool}` with `answer` set to your answer.
        - Otherwise simply repeat the number
        """,
    )
)

odd_agent = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="Odd",
        system_message=f"""
        - If you receive an odd number N, return 3N+1 using the
          TOOL `{my_final_result_tool}` with `answer` set to your answer.
        - Otherwise simply repeat the number        
        """,
    )
)


even_agent.enable_message(MyFinalResultTool)
odd_agent.enable_message(MyFinalResultTool)

# set up main_task to return a result of type MyFinalResultTool
main_task = lr.Task(main_agent, interactive=False)[MyFinalResultTool]
even_task = lr.Task(even_agent, interactive=False)
odd_task = lr.Task(odd_agent, interactive=False)
main_task.add_sub_task([even_task, odd_task])

result = main_task.run("3")
assert isinstance(result, MyFinalResultTool)
assert result.answer == 10

result = main_task.run("4")
assert isinstance(result, MyFinalResultTool)
assert result.answer == 2

# we can also pass in an integer value

result = main_task.run(15)
assert isinstance(result, MyFinalResultTool)
assert result.answer == 46

result = main_task.run(16)
assert isinstance(result, MyFinalResultTool)
assert result.answer == 8
