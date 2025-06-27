"""
Toy example where 3 agents concurrently respond to the current message,
and the current message is updated to the response of one such responder.

Run like this:

python3 examples/basic/multi-agent-round-table.py

"""

import langroid as lr
from langroid.agent.batch import run_batch_task_gen
from langroid.utils.constants import NO_ANSWER

agent1 = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="agent1",
        system_message=f"""
        You are a simple number transformer, follow this rule:
        - If you see a number ending in 0,1, or 2, respond with a random 3-digit number.
        - Otherwise, respond saying: {NO_ANSWER}
        """,
    )
)
task1 = lr.Task(agent1, interactive=False, single_round=True)

agent2 = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="agent2",
        system_message=f"""
        You are a simple number transformer, follow this rule:
        - If you see a number ending in 3,4, or 5, respond with a random 3-digit number.
        - Otherwise, respond saying: {NO_ANSWER}
        """,
    )
)
task2 = lr.Task(agent2, interactive=False, single_round=True)


agent3 = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="agent3",
        system_message=f"""
        You are a simple number transformer, follow this rule:
        - If you see a number ending in 6,7,8 or 9, respond with a random 3-digit number.
        - Otherwise, respond saying: {NO_ANSWER}
        """,
    )
)
task3 = lr.Task(agent3, interactive=False, single_round=True)

tasks = [task1, task2, task3]


def task_gen(i):
    return tasks[i]


# kickoff with n = 412
n = 412
# run for 10 rounds
for _ in range(10):
    print("n = ", n)
    inputs = [n] * 3
    results = run_batch_task_gen(task_gen, inputs)
    # find which result is not NO_ANSWER
    for i, r in enumerate(results):
        if r.content != NO_ANSWER:
            n = int(r.content)
            print(f"agent{i+1} responded with {n}")
            break
