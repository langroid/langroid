"""
Other tests for Task are in test_chat_agent.py
"""

import asyncio
from typing import List, Optional

import pytest

import langroid as lr
from langroid.agent import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import (
    AgentDoneTool,
    DonePassTool,
    DoneTool,
    PassTool,
)
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import Entity
from langroid.utils.configuration import (
    Settings,
    set_global,
    settings,
    temporary_settings,
)
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


@pytest.mark.parametrize("restart", [True, False])
def test_task_restart(test_settings: Settings, restart: bool):
    """Test whether the `restart` option works as expected"""
    set_global(test_settings)
    agent = ChatAgent(
        ChatAgentConfig(
            name="Test",
            llm=MockLMConfig(response_fn=lambda x: int(x) + 1),  # increment
        ),
    )
    task = Task(
        agent,
        interactive=False,
        single_round=False,
        restart=restart,
    )
    task.run("4", turns=1)  # msg hist = sys, user=4, asst=5
    # if restart, erases agent history => msg hist = sys, user=10, asst=11
    # otherwise, adds to msg history => msg hist = sys, user=4, asst=5, user=10, asst=11
    task.run("10", turns=1)
    if restart:
        assert len(agent.message_history) == 3
    else:
        assert len(agent.message_history) == 5


@pytest.mark.asyncio
async def test_task_kill(test_settings: Settings):
    """Test that Task.run() can be killed"""
    set_global(test_settings)

    class MockAgent(ChatAgent):
        async def llm_response_async(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument]:
            # dummy deterministic response; no need for real LLM here!
            return self.create_llm_response("hello")

    agent = MockAgent(ChatAgentConfig(name="Test"))
    with temporary_settings(Settings(max_turns=-1)):

        task = Task(
            agent,
            interactive=False,
            single_round=False,
            default_human_response="ok",
            config=lr.TaskConfig(inf_loop_cycle_len=0),  # turn off cycle detection
        )
        # start task
        async_task = asyncio.create_task(
            task.run_async("hi", turns=50, session_id="mysession")
        )
        # sleep a bit then kill it
        await asyncio.sleep(0.1)
        task.kill()
        result: lr.ChatDocument = await async_task
        assert result.metadata.status == lr.StatusCode.KILL

        # test killing via static method:
        # Run it for a potentially very large number of turns...
        async_task = asyncio.create_task(
            task.run_async("hi", turns=50, session_id="mysession")
        )
        # ...sleep a bit then kill it
        await asyncio.sleep(0.1)
        Task.kill_session("mysession")
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
@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("use_orch_tools", [True, False])
@pytest.mark.parametrize(
    "agent_done_pass",
    [True, False],
)
def test_task_tool_agent_response(
    test_settings: Settings,
    use_fn_api: bool,
    use_tools_api: bool,
    agent_done_pass: bool,
    use_orch_tools: bool,
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

        def handle(self) -> str | ToolMessage:
            if use_orch_tools:
                return DonePassTool() if agent_done_pass else DoneTool()
            else:
                return DONE + " " + PASS if agent_done_pass else DONE

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
            use_tools_api=use_tools_api,
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
        return isinstance(agent.get_tool_messages(response)[0], AugmentTool)

    def tool_valid():
        return "next_num" in response.content

    def fn_or_tool_valid():
        return fn_call_valid() if use_fn_api else tool_valid()

    if agent_done_pass:
        assert fn_or_tool_valid()
    else:
        assert content_empty()


@pytest.mark.parametrize("use_fn_api", [False, True])
@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("agent_response_done", [False, True])
@pytest.mark.parametrize("use_orch_tools", [True, False])
def test_task_tool_num(
    test_settings: Settings,
    use_fn_api: bool,
    use_tools_api: bool,
    agent_response_done: bool,
    use_orch_tools: bool,
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

        def handle(self) -> str | DoneTool:
            if agent_response_done:
                if use_orch_tools:
                    return DoneTool(content=str(self.successor))
                else:
                    return DONE + " " + str(self.successor)
            else:
                return str(self.successor)

    tool_name = AugmentTool.default_value("request")
    done_pass_tool_name = DonePassTool.default_value("request")
    if use_orch_tools:
        done_response = f"use the TOOL: `{done_pass_tool_name}`"
    else:
        done_response = f"say {DONE} {PASS}"

    agent = ChatAgent(
        ChatAgentConfig(
            name="Test",
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
            use_tools=not use_fn_api,
            system_message=f"""
            User will send a number. Augment it with its successor,
            and present the numbers using the `{tool_name}` tool/function.
            You will then receive a number as response.
            When you receive this, 
            {done_response}
            to signal that you are done, and that the result is the number you received.
            """,
        )
    )
    agent.enable_message(AugmentTool)
    agent.enable_message(DonePassTool)
    task = Task(
        agent,
        interactive=False,
        done_if_no_response=[Entity.LLM],
    )

    response = task.run("100")
    assert "101" in response.content


@pytest.mark.parametrize("use_fn_api", [True, False])
@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("use_orch_tools", [True, False])
def test_task_2_agent_tool(
    test_settings: Settings,
    use_fn_api: bool,
    use_tools_api: bool,
    use_orch_tools: bool,
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

        def polinsky(self, msg: PolinskyTool) -> str | PassTool:
            # No validation err, so pass it on
            if use_orch_tools:
                return PassTool()
            else:
                return PASS

    done_pass_tool_name = DonePassTool.default_value("request")
    if use_orch_tools:
        done_response = f"use the TOOL: `{done_pass_tool_name}`"
    else:
        done_response = f"say {DONE} {PASS}"
    requestor_agent = Requestor(
        ChatAgentConfig(
            name="Requestor",
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
            use_tools=not use_fn_api,
            system_message=f"""
                User will send a number. Your job is to find out what is
                the "Polinsky transform", which you KNOW is POSITIVE 
                but do not know how to compute,
                so you must use the `polinsky` tool/function to request it.
                When you get a POSITIVE value, 
                {done_response}
                to signal you're done.
                If you get a NEGATIVE value, you must AGAIN request the Polinsky
                of the ORIGINAL number, until you get a POSITIVE value.
                """,
        )
    )
    requestor_agent.enable_message(DonePassTool)
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
            use_tools_api=use_tools_api,
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


@pytest.mark.parametrize("use_fn_api", [False, True])
@pytest.mark.parametrize("use_tools_api", [True, False])
@pytest.mark.parametrize("use_orch_tools", [False, True])
def test_task_2_agent_2_tool(
    test_settings: Settings,
    use_fn_api: bool,
    use_tools_api: bool,
    use_orch_tools: bool,
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
            self.enable_message(DoneTool)

        def polinsky_query(self, msg: QueryTool) -> str | PassTool:
            # No validation err, so pass it on so other agent can respond
            return PassTool() if use_orch_tools else PASS

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

    done_tool_name = DoneTool.default_value("request")
    requestor_agent = Requestor(
        ChatAgentConfig(
            name="Requestor",
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
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
                    use the `{done_tool_name}` with `content` showing summary 
                    of the transforms in this format:
                    '(number1, transform1), (number2, transform2)'
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

        def polinsky_feedback(self, msg: FeedbackTool) -> str | DonePassTool:
            """Pass on the feedback to the Requestor"""
            return DonePassTool() if use_orch_tools else DONE + " " + PASS

    critic_agent = Critic(
        ChatAgentConfig(
            name="Critic",
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
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


def test_task_tool_responses():
    """Test that returning ToolMessage from an entity-responder or a Task.run() are
    handled correctly"""

    class IncrementTool(ToolMessage):
        request = "increment"
        purpose = "To increment a number"
        x: int

        def handle(self) -> str:
            return DoneTool(content=str(self.x + 1))

    class AnswerTool(ToolMessage):
        request = "answer"
        purpose = "To provide the final answer"
        answer: int

    class DoubleTool(ToolMessage):
        request = "double"
        purpose = "To double a number"
        x: int

        def handle(self) -> str:
            # return this as the double_task's answer
            return AgentDoneTool(tools=[AnswerTool(answer=2 * self.x)])

    class HalveTool(ToolMessage):
        request = "halve"
        purpose = "To halve a number"
        x: int

        def handle(self) -> str:
            return DoneTool(content=str(self.x // 2))

    class ProcessTool(ToolMessage):
        request = "process"
        purpose = "To process a number"
        x: int

        def handle(self) -> ToolMessage:
            if self.x % 10 == 0:
                return IncrementTool(x=self.x)
            elif self.x % 2 == 0:
                return HalveTool(x=self.x)
            else:
                return DoubleTool(x=self.x)

    class ProcessorAgent(lr.ChatAgent):
        def init_state(self):
            super().init_state()
            self.expecting_result: bool = False

        def llm_response(
            self, message: Optional[str | ChatDocument] = None
        ) -> Optional[ChatDocument | ToolMessage]:
            # return a ToolMessage rather than ChatDocument
            msg_str = message.content if isinstance(message, ChatDocument) else message
            if self.expecting_result:
                if msg_str != "":
                    return DoneTool(content=msg_str)
                elif (
                    isinstance(message, ChatDocument)
                    and len(message.tool_messages) > 0
                    and isinstance(message.tool_messages[0], AnswerTool)
                ):
                    # must be AnswerTool
                    answer_tool: AnswerTool = message.tool_messages[0]
                    return DoneTool(content=answer_tool.answer)
                else:
                    return None

            x = int(msg_str)
            self.expecting_result = True
            return ProcessTool(x=x)

    processor_agent = ProcessorAgent(lr.ChatAgentConfig(name="Processor"))
    processor_agent.enable_message(ProcessTool)
    processor_task = Task(processor_agent, interactive=False, restart=True)

    halve_agent = lr.ChatAgent(lr.ChatAgentConfig(name="Halver", llm=None))
    halve_agent.enable_message(HalveTool, use=False, handle=True)
    halve_agent.enable_message(IncrementTool, use=False, handle=False)
    halve_agent.enable_message(DoubleTool, use=False, handle=False)

    halve_task = Task(halve_agent, interactive=False)

    double_agent = lr.ChatAgent(lr.ChatAgentConfig(name="Doubler", llm=None))
    double_agent.enable_message(DoubleTool, use=False, handle=True)
    double_task = Task(double_agent, interactive=False)

    increment_agent = lr.ChatAgent(lr.ChatAgentConfig(name="Incrementer", llm=None))
    increment_agent.enable_message(IncrementTool, use=False, handle=True)
    increment_agent.enable_message(DoubleTool, use=False, handle=False)
    increment_task = Task(increment_agent, interactive=False)

    processor_task.add_sub_task([halve_task, increment_task, double_task])

    result = processor_task.run(str(3))
    assert result.content == str(6)

    # note: processor_agent state gets reset each time we run the task
    result = processor_task.run(str(16))
    assert result.content == str(8)

    result = processor_task.run(str(10))
    assert result.content == str(11)
