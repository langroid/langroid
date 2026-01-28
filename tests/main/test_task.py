"""
Other tests for Task are in test_chat_agent.py
"""

import asyncio
import json
from typing import Any, List, Optional

import pytest

import langroid as lr
from langroid.agent import ChatDocument
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task, TaskConfig
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import (
    AgentDoneTool,
    DonePassTool,
    DoneTool,
    PassTool,
)
from langroid.language_models.base import LLMMessage
from langroid.language_models.mock_lm import MockLMConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
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
        request: str = "next_num"
        purpose: str = """
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
@pytest.mark.parametrize("use_orch_tools", [False, True])
@pytest.mark.parametrize("string_signals", [False, True])
def test_task_tool_num(
    test_settings: Settings,
    use_fn_api: bool,
    use_tools_api: bool,
    agent_response_done: bool,
    use_orch_tools: bool,
    string_signals: bool,
):
    """
    Test loop within single agent, where this cycle repeats:
        [ LLM --Tool--> Agent[Tool] ---> (User) ]*

    The Agent responds to the tool with a number.
    """
    set_global(test_settings)

    class AugmentTool(ToolMessage):
        request: str = "next_num"
        purpose: str = """
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
    if use_orch_tools:
        agent.enable_message(DonePassTool)
    else:
        agent.disable_message_use(DonePassTool)
    task_config = TaskConfig(recognize_string_signals=string_signals)
    task = Task(
        agent,
        interactive=False,
        done_if_no_response=[Entity.LLM],
        config=task_config,
    )

    response = task.run("100", turns=10)
    if use_orch_tools or string_signals:
        assert "101" in response.content
        assert len(agent.message_history) <= 5
    else:
        # no orch tool, and string signals ignored, so task doesn't terminate,
        # and is limited by "turns" parameter
        assert len(agent.message_history) > 7


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
        request: str = "polinsky"
        purpose: str = """
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
@pytest.mark.parametrize("use_orch_tools", [True, False])
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
        request: str = "polinsky_query"
        purpose: str = """
            Ask whether the Polinsky transform of a <number> equals <value>.
            """
        number: int
        value: int

    class FeedbackTool(ToolMessage):
        request: str = "polinsky_feedback"
        purpose: str = """
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
            if msg.feedback.strip() == "":
                return f"""
                CORRECT, the value you gave IS the Polinsky transform of that number.
                Please proceed with requesting the Polinsky transform of 
                the NEXT number on your list, or if you're finished, use the
                TOOL `{DoneTool.name()}` with `content` set to the summary of the
                transformations, in this format:
                '(number1, transform1), (number2, transform2)'
                """
            else:
                return f"""
                WRONG, please try again based on this feedback: 
                {msg.feedback}
                """

        def handle_message_fallback(self, msg: str | ChatDocument) -> Any:
            if isinstance(msg, ChatDocument) and msg.metadata.sender == Entity.LLM:
                return f"""
                Your INTENT is unclear!
                
                - If you intended to say you're finished with your task,
                then use the `{DoneTool.name()}` tool/function with 
                the `content` field set to the summary of the Polinsky transforms
                of 100 and 500.
                
                - If you intended to ask about the Polinsky transform,
                then use the `{QueryTool.name()}` tool/function to ask about
                the Polinsky transform of a number.
                """

    done_tool_name = DoneTool.default_value("request")
    requestor_agent = Requestor(
        ChatAgentConfig(
            name="Requestor",
            allow_multiple_tools=False,
            use_functions_api=use_fn_api,
            use_tools_api=use_tools_api,
            use_tools=not use_fn_api,
            system_message=f"""
                    Your mission is to find the "Polinsky transform" of TWO NUMBERS:
                    100 and 500.
                    This is a mysterious transform that you do not
                    know how to compute, but you try to find out, by GUESSING the
                    value and asking for CONFIRMATION, 
                    using the `polinsky_query` tool/function, ONE NUMBER AT A TIME.
                    
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
                    
                    IMPORTANT - YOU CAN ONLY use the `polinsky_feedback` tool/function
                    ONCE per message.
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
            handle_llm_no_tool=f"you forgot to use the TOOL `{FeedbackTool.name()}`",
            system_message="""
            When you receive a query asking whether the Polinsky
            transform of a number x is y, and you must give FEEDBACK
            on this using the `polinsky_feedback` tool/function.
            Here are the rules:

            - If y = x + 1, use the `polinsky_feedback` tool 
              with `feedback` field = "WRONG, try another guess",

            - Otherwise, use the `polinsky_feedback` tool 
              with `feedback` field set to an EMPTY STRING: ""
            
            IMPORTANT - YOU CAN ONLY use the `polinsky_feedback` tool/function
            ONCE per message.
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


def test_task_tool_responses(
    test_settings: Settings,
):
    """Test that returning ToolMessage from an entity-responder or a Task.run() are
    handled correctly"""

    set_global(test_settings)

    class IncrementTool(ToolMessage):
        request: str = "increment"
        purpose: str = "To increment a number"
        x: int

        def handle(self) -> str:
            return DoneTool(content=str(self.x + 1))

    class AnswerTool(ToolMessage):
        request: str = "answer"
        purpose: str = "To provide the final answer"
        answer: int

    class DoubleTool(ToolMessage):
        request: str = "double"
        purpose: str = "To double a number"
        x: int

        def handle(self) -> str:
            # return this as the double_task's answer
            return AgentDoneTool(tools=[AnswerTool(answer=2 * self.x)])

    class HalveTool(ToolMessage):
        request: str = "halve"
        purpose: str = "To halve a number"
        x: int

        def handle(self) -> str:
            return DoneTool(content=self.x // 2)  # note: content can be any type

    class ProcessTool(ToolMessage):
        request: str = "process"
        purpose: str = "To process a number"
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

    result = processor_task[int].run(10)
    assert result == 11


def test_task_output_format_sequence():
    """
    Test that `Task`s correctly execute a sequence of steps
    controlled by the agent's `output_format`, and that `output_format`
    is handled by default without `enable_message`.
    """

    class MultiplyTool(ToolMessage):
        request: str = "multiply"
        purpose: str = "To multiply two integers."
        a: int
        b: int

    class IncrementTool(ToolMessage):
        request: str = "increment"
        purpose: str = "To increment an integer."
        x: int

    class PowerTool(ToolMessage):
        request: str = "power"
        purpose: str = "To compute `x` ** `y`."
        x: int
        y: int

    class CompositionAgent(ChatAgent):
        def __init__(self, config: ChatAgentConfig = ChatAgentConfig()):
            super().__init__(config)
            self.set_output_format(MultiplyTool)

        def multiply(self, message: MultiplyTool) -> str:
            self.set_output_format(IncrementTool)

            return str(message.a * message.b)

        def increment(self, message: IncrementTool) -> str:
            self.set_output_format(PowerTool)

            return str(message.x + 1)

        def power(self, message: PowerTool) -> str:
            return f"{DONE} {message.x ** message.y}"

    def to_tool(message: LLMMessage, tool: type[ToolMessage]) -> ToolMessage:
        return tool.model_validate(json.loads(message.content))

    def test_sequence(x: int) -> None:
        agent = CompositionAgent(
            ChatAgentConfig(
                llm=OpenAIGPTConfig(
                    supports_json_schema=True,
                    supports_strict_tools=True,
                ),
            )
        )
        task = lr.Task(
            agent,
            system_message="""
            You will be provided with a number `x` and will compute (3 * x + 1) ** 4,
            using these ops sequentially: 
            - multiplication, to compute 3*x to get result M, using the `multiply` tool
            - increment, to compute M + 1 to get result N, using the `increment` tool
            - power, to compute N ** 4 to get result P, using the `power` tool
            """,
            interactive=False,
            default_return_type=int,
        )
        output = task.run(x)
        assert isinstance(output, int)
        assert output == (3 * x + 1) ** 4

        # check steps
        messages = agent.message_history
        assert len(messages) >= 7

        multiply_message: MultiplyTool = to_tool(messages[2], MultiplyTool)  # type: ignore
        assert {multiply_message.a, multiply_message.b} == {3, x}

        increment_message: IncrementTool = to_tool(messages[4], IncrementTool)  # type: ignore
        assert increment_message.x == 3 * x

        power_message: PowerTool = to_tool(messages[6], PowerTool)  # type: ignore
        assert (power_message.x, power_message.y) == (3 * x + 1, 4)

    for x in range(5):
        test_sequence(x)


@pytest.mark.asyncio
async def test_task_output_format_sequence_async():
    """
    Test that async `Task`s correctly execute a sequence of steps
    controlled by the agent's `output_format`, and that `output_format`
    is handled by default without `enable_message`.
    """

    class MultiplyTool(ToolMessage):
        request: str = "multiply"
        purpose: str = "To multiply two integers."
        a: int
        b: int

    class IncrementTool(ToolMessage):
        request: str = "increment"
        purpose: str = "To increment an integer."
        x: int

    class PowerTool(ToolMessage):
        request: str = "power"
        purpose: str = "To compute `x` ** `y`."
        x: int
        y: int

    class CompositionAgent(ChatAgent):
        def __init__(self, config: ChatAgentConfig = ChatAgentConfig()):
            super().__init__(config)
            self.set_output_format(MultiplyTool)

        def multiply(self, message: MultiplyTool) -> str:
            self.set_output_format(IncrementTool)

            return str(message.a * message.b)

        def increment(self, message: IncrementTool) -> str:
            self.set_output_format(PowerTool)

            return str(message.x + 1)

        def power(self, message: PowerTool) -> str:
            self.set_output_format(MultiplyTool)

            return f"{DONE} {message.x ** message.y}"

    def to_tool(message: LLMMessage, tool: type[ToolMessage]) -> ToolMessage:
        return tool.model_validate(json.loads(message.content))

    async def test_sequence(x: int) -> None:
        agent = CompositionAgent(
            ChatAgentConfig(
                llm=OpenAIGPTConfig(
                    supports_json_schema=True,
                    supports_strict_tools=True,
                ),
            )
        )
        task = lr.Task(
            agent,
            system_message="""
            You will be provided with a number `x` and will compute (3 * x + 1) ** 4,
            using these ops sequentially: 
            - multiplication, to compute 3*x to get result M, using the `multiply` tool
            - increment, to compute M + 1 to get result N, using the `increment` tool
            - power, to compute N ** 4 to get result P, using the `power` tool
            """,
            interactive=False,
            default_return_type=int,
        )
        output = await task.run_async(x)
        assert isinstance(output, int)
        assert output == (3 * x + 1) ** 4

        # check steps
        messages = agent.message_history
        assert len(messages) >= 7

        multiply_message: MultiplyTool = to_tool(messages[2], MultiplyTool)  # type: ignore
        assert {multiply_message.a, multiply_message.b} == {3, x}

        increment_message: IncrementTool = to_tool(messages[4], IncrementTool)  # type: ignore
        assert increment_message.x == 3 * x

        power_message: PowerTool = to_tool(messages[6], PowerTool)  # type: ignore
        assert (power_message.x, power_message.y) == (3 * x + 1, 4)

    for x in range(5):
        await test_sequence(x)


def test_done_if_tool(test_settings: Settings):
    """Test that task terminates when LLM generates a tool and done_if_tool=True"""

    set_global(test_settings)

    class SimpleTool(ToolMessage):
        request: str = "simple_tool"
        purpose: str = "A simple tool for testing"
        value: str = "test"

    # Create a mock LLM that always responds with a tool in JSON format
    tool_response = SimpleTool(value="hello").to_json()

    mock_lm_config = MockLMConfig(default_response=tool_response)

    # Create agent with mock LLM
    agent = ChatAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=mock_lm_config,
        )
    )

    # Enable the tool but don't handle it
    # (the `use` doesn't matter here since we're hard-coding the MockLM response
    # to always be the tool-call)
    agent.enable_message(SimpleTool, use=False, handle=False)

    # Test 1: With done_if_tool=False (default), task should not terminate on tool
    task = Task(
        agent,
        interactive=False,
        config=TaskConfig(done_if_tool=False),
    )

    result = task.run("Process this message", turns=3)
    # Task should run for all 3 turns since done_if_tool=False
    assert result is not None
    assert len(agent.message_history) >= 3  # At least: system, user, assistant

    # Reset agent for next test
    agent.clear_history()

    # Test 2: With done_if_tool=True, task should terminate when tool is generated
    task_with_done = Task(
        agent,
        interactive=False,
        config=TaskConfig(done_if_tool=True),
    )

    result = task_with_done.run("Do something else", turns=10)
    # Task should terminate after first LLM response containing tool
    assert result is not None
    # Should have exactly 3 messages: system, user, assistant (with tool)
    assert len(agent.message_history) == 3

    # Verify the last message contains the tool
    last_msg = agent.message_history[-1]
    assert "simple_tool" in last_msg.content

    # Reset agent for next test
    agent.clear_history()

    # Test 3: With done_if_tool=True and return type specified
    task_with_return_type = Task(
        agent,
        interactive=False,
        config=TaskConfig(done_if_tool=True),
    )[SimpleTool]

    result_typed = task_with_return_type.run("Process with type", turns=10)
    # Task should terminate and return the SimpleTool instance
    assert result_typed is not None
    assert isinstance(result_typed, SimpleTool)
    assert result_typed.value == "hello"


def test_task_init_preserves_parent_id():
    """Test that task.init() preserves parent_id when deep copying ChatDocument"""

    # Create an agent
    agent = ChatAgent(ChatAgentConfig(name="TestAgent"))

    # Test 1: Basic parent_id preservation
    parent_doc = ChatDocument(
        content="Parent message",
        metadata=lr.agent.chat_document.ChatDocMetaData(
            sender=Entity.USER,
        ),
    )
    child_doc = ChatDocument(
        content="Child message",
        metadata=lr.agent.chat_document.ChatDocMetaData(
            parent_id=parent_doc.id(),
            sender=Entity.USER,
        ),
    )

    task = Task(agent, interactive=False)
    task.init(child_doc)

    # The pending message should preserve the parent_id
    assert task.pending_message is not None
    assert task.pending_message.metadata.parent_id == parent_doc.id()
    assert task.pending_message.content == "Child message"

    # Test 2: With caller (subtask scenario)
    caller_task = Task(agent, interactive=False, name="CallerTask")
    sub_task = Task(agent, interactive=False, name="SubTask")

    # When message already has parent_id, it should be preserved
    msg_with_parent = ChatDocument(
        content="Message with parent",
        metadata=lr.agent.chat_document.ChatDocMetaData(
            parent_id=parent_doc.id(),
            sender=Entity.USER,
        ),
    )

    # Set caller to simulate subtask scenario
    sub_task.caller = caller_task
    sub_task.init(msg_with_parent)

    # Parent_id should still be preserved (not overridden to msg.id)
    assert sub_task.pending_message is not None
    assert sub_task.pending_message.metadata.parent_id == parent_doc.id()

    # Test 3: With caller but no original parent_id
    msg_no_parent = ChatDocument(
        content="Message without parent",
        metadata=lr.agent.chat_document.ChatDocMetaData(
            sender=Entity.USER,
        ),
    )

    sub_task2 = Task(agent, interactive=False, name="SubTask2")
    sub_task2.caller = caller_task
    sub_task2.init(msg_no_parent)

    # Since original had no parent_id, it should be set to msg.id
    assert sub_task2.pending_message is not None
    assert sub_task2.pending_message.metadata.parent_id == msg_no_parent.id()


@pytest.mark.parametrize("recognize_recipient", [True, False])
def test_recognize_recipient_in_content(
    test_settings: Settings,
    recognize_recipient: bool,
):
    """
    Test that ChatAgentConfig.recognize_recipient_in_content controls whether
    recipient patterns (TO[recipient]:message) are parsed from LLM responses.

    When True (default): recipient is extracted, content has pattern removed
    When False: recipient is empty, content preserves the TO[...] pattern
    """
    set_global(test_settings)

    # LLM response containing recipient pattern
    llm_response_text = "TO[SubAgent]: Please handle this request"

    agent = ChatAgent(
        ChatAgentConfig(
            name="Test",
            llm=MockLMConfig(
                default_response=llm_response_text,
            ),
            recognize_recipient_in_content=recognize_recipient,
        )
    )

    task = Task(agent, interactive=False, single_round=True)
    result = task.run("Hello")

    if recognize_recipient:
        # Recipient should be extracted
        assert result.metadata.recipient == "SubAgent"
        # Content should have pattern stripped
        assert "TO[" not in result.content
        assert "Please handle this request" in result.content
    else:
        # Recipient should NOT be extracted
        assert result.metadata.recipient == ""
        # Content should preserve the original pattern
        assert "TO[SubAgent]:" in result.content


@pytest.mark.parametrize("recognize_recipient", [True, False])
def test_recognize_recipient_json_format(
    test_settings: Settings,
    recognize_recipient: bool,
):
    """
    Test that recognize_recipient_in_content also controls parsing of
    JSON recipient format: {"recipient": "...", ...}
    """
    set_global(test_settings)

    # LLM response with JSON recipient pattern
    llm_response_text = '{"recipient": "SubAgent", "content": "Handle this"}'

    agent = ChatAgent(
        ChatAgentConfig(
            name="Test",
            llm=MockLMConfig(
                default_response=llm_response_text,
            ),
            recognize_recipient_in_content=recognize_recipient,
        )
    )

    task = Task(agent, interactive=False, single_round=True)
    result = task.run("Hello")

    if recognize_recipient:
        # Recipient should be extracted from JSON
        assert result.metadata.recipient == "SubAgent"
    else:
        # Recipient should NOT be extracted
        assert result.metadata.recipient == ""
        # Content should preserve the original JSON
        assert '"recipient"' in result.content
