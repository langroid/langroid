"""
Other tests for Task are in test_chat_agent.py
"""

import asyncio
from typing import List

import pytest

import langroid as lr
from langroid.agent import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.mytypes import Entity
from langroid.utils.configuration import Settings, set_global, settings
from langroid.utils.constants import DONE, PASS


def test_task_cost(test_settings: Settings):
    """Test that max_cost, max_tokens are respected by Task.run()"""

    set_global(test_settings)
    settings.cache = False
    agent = ChatAgent(ChatAgentConfig(name="Test"))
    agent.llm.reset_usage_cost()
    task = Task(
        agent,
        interactive=False,
        single_round=False,
        system_message="User will send you a number. Repeat the number.",
    )
    sub_agent = ChatAgent(ChatAgentConfig(name="Sub"))
    sub_agent.llm.reset_usage_cost()
    sub = Task(
        sub_agent,
        interactive=False,
        single_round=True,
        system_message="User will send you a number. Return its double",
    )
    task.add_sub_task(sub)
    response = task.run("4", turns=10, max_cost=0.0005, max_tokens=100)
    settings.cache = True
    assert response is not None
    assert response.metadata.status in [
        lr.StatusCode.MAX_COST,
        lr.StatusCode.MAX_TOKENS,
    ]


@pytest.mark.asyncio
async def test_task_kill(test_settings: Settings):
    """Test that Task.run() can be killed"""
    set_global(test_settings)
    agent = ChatAgent(ChatAgentConfig(name="Test"))
    task = Task(
        agent,
        interactive=False,
        single_round=False,
        default_human_response="double your last number",
        system_message="User will send you a number. Return its double.",
    )
    # start task
    async_task = asyncio.create_task(task.run_async("2", turns=20))
    # sleep a bit then kill it
    await asyncio.sleep(1)
    task.kill()
    result: lr.ChatDocument = await async_task
    assert result.metadata.status == lr.StatusCode.KILL


def test_task_empty_response(test_settings: Settings):
    set_global(test_settings)
    agent = ChatAgent(ChatAgentConfig(name="Test"))
    task = Task(
        agent,
        interactive=False,
        done_if_response=[Entity.LLM],
        done_if_no_response=[Entity.LLM],
        system_message="""
        User will send you a number. 
        If it is EVEN, repeat the number, else return empty string.
        ONLY return these responses, say NOTHING ELSE
        """,
    )

    response = task.run("4")
    assert response.content == "4"
    response = task.run("3")
    assert response.content == ""


@pytest.mark.parametrize(
    "even_response, odd_response, "
    "done_if_response, done_if_no_response, "
    "even_result, odd_result",
    [
        (f"say '{DONE} {PASS}'", f"say {DONE}", [], [], "4", ""),
        (
            "repeat the number",
            "return empty string",
            [Entity.LLM],
            [Entity.LLM],
            "4",
            "",
        ),
        (
            f"say '{DONE} {PASS}'",
            "return empty string",
            [],
            [lr.mytypes.Entity.LLM],
            "4",
            "",
        ),
    ],
)
def test_task_done_condition(
    test_settings: Settings,
    even_response: str,
    odd_response: str,
    done_if_response: List[str],
    done_if_no_response: List[str],
    even_result: str,
    odd_result: str,
):
    set_global(test_settings)

    # test done_if_response, done_if_no_response
    agent = ChatAgent(ChatAgentConfig(name="Test"))
    task = Task(
        agent,
        interactive=False,
        done_if_response=done_if_response,
        done_if_no_response=done_if_no_response,
        system_message=f"""
        User will send you a number. 
        If it is EVEN, {even_response}, 
        Otherwise {odd_response}.
        ONLY return these responses, say NOTHING ELSE
        """,
    )

    response = task.run("4")
    assert response.content == even_result
    response = task.run("3")
    assert response.content == odd_result


@pytest.mark.parametrize(
    "default_human_response, sys_msg, input, expected",
    [
        (PASS, "User gives a number, return its double", "4", "8"),
        (
            "HELLO",
            "User gives a number, return its double",
            "4",
            "8",
        ),
        (
            "",
            "Whatever user says, you return empty string",
            "4",
            "",
        ),
        (
            "",
            f"Whatever user says, you say {DONE}",
            "4",
            "",
        ),
    ],
)
def test_task_default_human_response(
    test_settings: Settings,
    default_human_response: str,
    sys_msg: str,
    input: str,
    expected: str,
):
    set_global(test_settings)
    agent = ChatAgent(ChatAgentConfig(name="Test"))
    task = Task(
        agent,
        interactive=False,
        done_if_response=[Entity.LLM],
        done_if_no_response=[Entity.LLM],
        default_human_response=default_human_response,
        system_message=sys_msg,
    )

    response = task.run(input)
    assert expected in response.content


@pytest.mark.parametrize("use_fn_api", [True, False])
@pytest.mark.parametrize(
    "agent_response",
    [f"{DONE} {PASS}", DONE],
)
def test_task_tool_agent_response(
    test_settings: Settings,
    use_fn_api: bool,
    agent_response: str,
):
    """
    Test loop within single agent, where this cycle repeats:
        [ LLM --Tool--> Agent[Tool] ---> (User) ]*

    Test expected behavior for various Agent-tool-handler responses.
    """
    set_global(test_settings)

    class AugmentTool(ToolMessage):
        request = "next_num"
        purpose = """
        To augment the given <number> with its <successor> = <number> + 1
        """
        number: int
        successor: int

        def handle(self) -> str:
            return agent_response

        @classmethod
        def examples(cls) -> List["ToolMessage"]:
            return [
                cls(
                    number=100,
                    successor=101,
                ),
            ]

        @staticmethod
        def handle_message_fallback(
            agent, msg: str | ChatDocument
        ) -> str | ChatDocument | None:
            if isinstance(msg, ChatDocument) and msg.metadata.sender == Entity.LLM:
                return """
                    You must use the `next_num` tool/function to 
                    augment the given number.
                    """
            return None

    agent = ChatAgent(
        ChatAgentConfig(
            name="Test",
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
            system_message="""
            User will send a number. Present this number and its successor,
            using the `next_num` tool/function.
            """,
        )
    )
    agent.enable_message(AugmentTool)
    task = Task(agent, interactive=False)

    response = task.run("100")

    def content_empty():
        return response.content == ""

    def fn_call_valid():
        return response.function_call.name == "next_num"

    def tool_valid():
        return "next_num" in response.content

    def fn_or_tool_valid():
        return fn_call_valid() if use_fn_api else tool_valid()

    match agent_response:
        case x if x == DONE:
            assert content_empty()
        case x if x == f"{DONE} {PASS}":
            assert fn_or_tool_valid()


@pytest.mark.parametrize("use_fn_api", [True, False])
@pytest.mark.parametrize("agent_response", ["{successor}", f"{DONE} {{successor}}"])
def test_task_tool_num(
    test_settings: Settings,
    use_fn_api: bool,
    agent_response: str,
):
    """
    Test loop within single agent, where this cycle repeats:
        [ LLM --Tool--> Agent[Tool] ---> (User) ]*

    The Agent responds to the tool with a number.
    """
    set_global(test_settings)

    class AugmentTool(ToolMessage):
        request = "next_num"
        purpose = """
        To augment the given <number> with its <successor> = <number> + 1
        """
        number: int
        successor: int

        def handle(self) -> str:
            return agent_response.format(
                number=self.number,
                successor=self.successor,
            )

    agent = ChatAgent(
        ChatAgentConfig(
            name="Test",
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
            system_message=f"""
            User will send a number. Augment it with its successor,
            and present the numbers using the `number` tool/function.
            You will then receive a number as response.
            When you receive this, simply say '{DONE} {PASS}'. 
            """,
        )
    )
    agent.enable_message(AugmentTool)
    task = Task(
        agent,
        interactive=False,
        done_if_no_response=[Entity.LLM],
    )

    response = task.run("100")
    assert "101" in response.content


@pytest.mark.parametrize("use_fn_api", [True, False])
def test_task_2_agent_tool(
    test_settings: Settings,
    use_fn_api: bool,
):
    """
    Test task loop where Agent B's task is a subtask of Agent A's task, and:
    - LLM(A) generates Tool
    - Agent(A) validates Tool, if error, LLM(A) needs to fix, else pass to Agent(B)
    - Agent(B) handles Tool, generates a query for LLM(B) to respond to
    - LLM(B) responds, result should be sent back to Agent(A)
    """

    set_global(test_settings)

    class PolinskyTool(ToolMessage):
        request = "polinsky"
        purpose = """
            Given a <number>, request its Polinsky transform.
            """
        number: int

    class Requestor(ChatAgent):
        def __init__(self, config: ChatAgentConfig):
            super().__init__(config)
            self.enable_message(PolinskyTool, use=True, handle=True)

        def polinsky(self, msg: PolinskyTool) -> str:
            # No validation err, so pass it on
            return PASS

    requestor_agent = Requestor(
        ChatAgentConfig(
            name="Requestor",
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
            system_message=f"""
                User will send a number. Your job is to find out what is
                the "Polinsky transform", which you KNOW is POSITIVE 
                but do not know how to compute,
                so you must use the `polinsky` tool/function to request it.
                When you get a POSITIVE value, simply say '{DONE} {PASS}'.
                If you get a NEGATIVE value, you must AGAIN request the Polinsky
                of the ORIGINAL number, until you get a POSITIVE value.
                """,
        )
    )
    requestor_task = Task(
        requestor_agent,
        interactive=False,
    )

    class PolinskyAgent(ChatAgent):
        def __init__(self, config: ChatAgentConfig):
            self.n_tries = 0
            super().__init__(config)
            self.enable_message(PolinskyTool, use=False, handle=True)

        def polinsky(self, msg: PolinskyTool) -> str:
            # Pass on the number so LLM can respond
            # On the first try, flip the sign of the number,
            # to force the Requestor to try again
            response = str(-msg.number) if self.n_tries == 0 else str(msg.number)
            self.n_tries += 1
            return response

    polinsky_agent = PolinskyAgent(
        ChatAgentConfig(
            name="Polinsky",
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
            system_message="""
                When you receive a number, respond with the DOUBLE of that number,
                say nothing else.
                """,
        )
    )
    polinsky_task = Task(
        polinsky_agent,
        interactive=False,
        # below ensure that task returns to requestor_task when LLM responds
        done_if_no_response=[Entity.LLM],
        done_if_response=[Entity.LLM],
    )

    # connect the two agents
    requestor_task.add_sub_task(polinsky_task)

    response = requestor_task.run("100")
    assert "200" in response.content


@pytest.mark.parametrize("use_fn_api", [True, False])
def test_task_2_agent_2_tool(
    test_settings: Settings,
    use_fn_api: bool,
):
    """
    QueryTool: Task A uses and handles (validates), Task B handles but doesn't use
    FeedbackTool: Task B uses and handles (validates), Task A handles but doesn't use
    """

    set_global(test_settings)

    class QueryTool(ToolMessage):
        request = "polinsky_query"
        purpose = """
            Ask whether the Polinsky transform of a <number> equals <value>.
            """
        number: int
        value: int

    class FeedbackTool(ToolMessage):
        request = "polinsky_feedback"
        purpose = """
            Given a <number>, respond with the Polinsky transform of the number.
            """
        feedback: str

    class Requestor(ChatAgent):
        def __init__(self, config: ChatAgentConfig):
            super().__init__(config)
            self.enable_message(QueryTool, use=True, handle=True)
            self.enable_message(FeedbackTool, use=False, handle=True)

        def polinsky_query(self, msg: QueryTool) -> str:
            # No validation err, so pass it on so other agent can respond
            return PASS

        def polinsky_feedback(self, msg: FeedbackTool) -> str:
            """Transmit feedback received from other agent, to this agent's LLM"""
            if msg.feedback == "":
                return """
                CORRECT, the value you gave IS the Polinsky transform of that number.
                Please proceed with requesting the Polinsky transform of 
                a NEW number.
                """
            else:
                return f"""
                WRONG, please try again based on this feedback: 
                {msg.feedback}
                """

    requestor_agent = Requestor(
        ChatAgentConfig(
            name="Requestor",
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
            system_message=f"""
                    Your mission is to find the "Polinsky transform" of TWO NUMBERS:
                    100 and 500.
                    This is a mysterious transform that you do not
                    know how to compute, but you try to find out, by GUESSING the
                    value and asking for CONFIRMATION, 
                    using the `polinsky_query` tool/function.
                    Your FIRST GUESS is to simply guess that the Polinsky transform
                    of your number is the SUCCESSOR of the number.
                    Your SECOND GUESS is to guess that the Polinsky transform
                    of your number is the DOUBLE of the number.
                    
                    You will receive feedback on your guess, and:
                    - if the feedback says "CORRECT", you can proceed with requesting
                        the Polinsky transform of the OTHER number.
                    - if the feedback says "WRONG", you must try again, using the
                        given feedback to guide your guess.
                        
                    When you have found out the Polinsky transform of 100 and 500,
                    say "{DONE} and summarize the transforms in this format:
                    'DONE (number1, transform1), (number2, transform2)'
                    """,
        )
    )
    requestor_task = Task(
        requestor_agent,
        interactive=False,
    )

    class Critic(ChatAgent):
        def __init__(self, config: ChatAgentConfig):
            super().__init__(config)
            self.enable_message(QueryTool, use=False, handle=True)
            self.enable_message(FeedbackTool, use=True, handle=True)

        def polinsky_query(self, msg: QueryTool) -> str:
            # pass on the number so LLM can respond
            return f"Is the Polinsky transform of {msg.number} equal to {msg.value}?"

        def polinsky_feedback(self, msg: FeedbackTool) -> str:
            """Pass on the feedback to the Requestor"""
            return DONE + " " + PASS

    critic_agent = Critic(
        ChatAgentConfig(
            name="Critic",
            use_functions_api=use_fn_api,
            use_tools=not use_fn_api,
            system_message="""
                    When you receive a query asking whether the Polinsky
                    transform of a number x is y, and you must give FEEDBACK
                    on this using the `polinsky_feedback` tool/function.
                    Here are the rules:
                    - If y = x + 1, feedback should be "WRONG, try another guess",
                    - Otherwise, feedback should be EMPTY STRING: ""
                    """,
        )
    )

    critic_task = Task(
        critic_agent,
        interactive=False,
    )

    # connect the two agents
    requestor_task.add_sub_task(critic_task)
    response = requestor_task.run()
    strings = "100 200 500 1000".split()
    assert all(s in response.content for s in strings)
