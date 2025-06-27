"""
Planner agent receives a math calculation expression from user,
involving + - * / ops, with possible parentheses. Planner has no math abilities,
so it needs to create a plan of elementary operations to compute the result,
and send each step to the appropriate helper agent, who will return the result.

Run like this:

python3 examples/basic/plan-subtasks.py

When it waits for user input, try asking things like:

- (10 + 2)/6 - 1
- 3*(4+1) - 3

"""

import langroid as lr
from langroid.utils.constants import AT, DONE, NO_ANSWER

planner = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="Planner",
        system_message=f"""
        User will give you a math calculation, but you have no math abilities.
        However you are a great planner, so your task is to do two things:
        
        1. CREATE a PLAN of what 
          sequence of ELEMENTARY operations (ONLY add/subtract, multiply/divide) need
          to performed, in order to compute what the user asked for.
        2. EMIT the needed operations, ONE BY ONE, and wait for the answer from
            each, before emitting the next operation. Since you cannot directly
            calculate these, you will have to SEND the needed operations to 
            specific helpers, as follows:
            
            * Send Multiplication operation to `Multiplier`
            * Send Add operation to `Adder`
            * Send Subtract operation to `Subtractor`
            * Send Divide operation to `Divider`
            
            To clarify who you are sending the message to, preface your message with
            {AT}<helper_name>, e.g. "{AT}Multiplier multiply with 5" 
            
            When you have the final answer, say {DONE} and show it.
            
            At the START, ask the user what they need help with, 
            address them as "{AT}user"
            
        EXAMPLE: 
        ============
        User: please calculate (4*5 + 1)/3
        Assistant (You): 
            PLAN: 
                1. multiply 4 with 5
                2. add 1 to the result
                3. divide result by 3
            {AT}Multiplier multiply 4 with 5
            [... wait for result, then show your NEW PLAN and send a new request]
            and so on.                         
                        
        """,
    )
)

adder = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="Adder",
        system_message=f"""
        If you receive an Add request, return the result,
        otherwise say {NO_ANSWER}.
        """,
    )
)

multiplier = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="Multiplier",
        system_message=f"""
        If you receive a Multiply request, return the result,
        otherwise say {NO_ANSWER}.
        """,
    )
)

subtractor = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="Subtractor",
        system_message=f"""
        If you receive a Subtraction request, return the result,
        otherwise say {NO_ANSWER}.
        """,
    )
)

divider = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="Divider",
        system_message=f"""
        If you receive a Division request, return the result,
        otherwise say {NO_ANSWER}.
        """,
    )
)


task_config = lr.TaskConfig(addressing_prefix=AT)
planner_task = lr.Task(planner, interactive=False, config=task_config)
adder_task = lr.Task(adder, interactive=False, single_round=True)
multiplier_task = lr.Task(multiplier, interactive=False, single_round=True)
divider_task = lr.Task(divider, interactive=False, single_round=True)
subtractor_task = lr.Task(subtractor, interactive=False, single_round=True)

planner_task.add_sub_task([adder_task, multiplier_task, divider_task, subtractor_task])


planner_task.run()
