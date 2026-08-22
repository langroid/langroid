"""
Tests for the pre-execution tool policy hook (`AgentConfig.tool_policy`),
issue #1095.

Contract:

1. No hook configured => tool dispatch behaves exactly as before (sync + async).
2. Hook allows (returns True or None) => the selected handler runs exactly once.
3. Hook denies (returns False or a str reason) => the handler runs zero times,
   and an LLM-visible rejection (tool name + reason) is produced.
4. The hook receives the final parsed ToolMessage (identity + payload) plus
   bounded context (the agent, and the chat_doc when available), BEFORE
   execution.
5. Hook failure (exception) => fail-closed rejection; raw tool arguments do
   NOT leak into the rejection message.
6. Composes with (does not weaken) the USER-origin tool security filter.

The hook must work on both sync and async dispatch paths, with both sync and
async callables. It cannot mutate what the handler executes, it is consulted
only when a real handler is selected, and it is shared by reference across
agent clones.
"""

import asyncio
import logging
import threading
from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Type, Union

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.mock_lm import MockLMConfig
from langroid.mytypes import Entity

HandleResult = Union[None, str, OrderedDict[str, str], ChatDocument]


class SquareTool(ToolMessage):
    request: str = "square"
    purpose: str = "To compute the square of a number <x>"
    x: int


class NoHandlerTool(ToolMessage):
    request: str = "orphan_tool"
    purpose: str = "A tool with no handler anywhere"
    y: int


class CountingAgent(ChatAgent):
    """Agent with a sync handler for SquareTool that counts invocations."""

    def __init__(self, config: ChatAgentConfig):
        super().__init__(config)
        self.calls: int = 0

    def square(self, msg: SquareTool) -> str:
        self.calls += 1
        return f"SQUARE_RESULT: {msg.x ** 2}"


class CountingAsyncAgent(CountingAgent):
    """Agent with a genuine async handler for SquareTool."""

    async def square_async(self, msg: SquareTool) -> str:
        self.calls += 1
        return f"SQUARE_RESULT: {msg.x ** 2}"


def mk_agent(
    tool_policy: Any = None,
    agent_cls: Type[CountingAgent] = CountingAgent,
) -> CountingAgent:
    agent = agent_cls(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(response_fn=lambda x: x),
            tool_policy=tool_policy,
        )
    )
    agent.enable_message(SquareTool)
    return agent


def llm_tool_msg(agent: ChatAgent, x: int = 3) -> ChatDocument:
    """An LLM-origin ChatDocument containing a SquareTool call."""
    return agent.create_llm_response(SquareTool(x=x).to_json())


def result_content(result: HandleResult) -> str:
    assert result is not None
    return result.content if isinstance(result, ChatDocument) else str(result)


# -------------------- contract 1: no hook => unchanged --------------------


def test_no_policy_sync_dispatch_unchanged() -> None:
    agent = mk_agent()
    assert agent.config.tool_policy is None
    result = agent.handle_message(llm_tool_msg(agent))
    assert "SQUARE_RESULT: 9" in result_content(result)
    assert agent.calls == 1


@pytest.mark.asyncio
async def test_no_policy_async_dispatch_unchanged() -> None:
    agent = mk_agent(agent_cls=CountingAsyncAgent)
    result = await agent.handle_message_async(llm_tool_msg(agent))
    assert "SQUARE_RESULT: 9" in result_content(result)
    assert agent.calls == 1


# -------------------- contract 2: allow => handler runs once --------------------


@pytest.mark.parametrize("decision", [True, None])
def test_allow_sync(decision: Optional[bool]) -> None:
    agent = mk_agent(
        tool_policy=lambda tool, agent: decision,
    )
    result = agent.handle_message(llm_tool_msg(agent, x=5))
    assert "SQUARE_RESULT: 25" in result_content(result)
    assert agent.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [True, None])
async def test_allow_async(decision: Optional[bool]) -> None:
    agent = mk_agent(
        tool_policy=lambda tool, agent: decision,
        agent_cls=CountingAsyncAgent,
    )
    result = await agent.handle_message_async(llm_tool_msg(agent, x=5))
    assert "SQUARE_RESULT: 25" in result_content(result)
    assert agent.calls == 1


# ---------------- contract 3: deny => zero runs, LLM-visible rejection ----------


def test_deny_with_reason_sync() -> None:
    agent = mk_agent(tool_policy=lambda tool, agent: "budget exceeded")
    result = agent.handle_message(llm_tool_msg(agent, x=5))
    content = result_content(result)
    assert agent.calls == 0
    assert "square" in content  # tool name
    assert "budget exceeded" in content  # policy's reason
    assert "SQUARE_RESULT" not in content


def test_deny_with_false_sync() -> None:
    agent = mk_agent(tool_policy=lambda tool, agent: False)
    result = agent.handle_message(llm_tool_msg(agent, x=5))
    content = result_content(result)
    assert agent.calls == 0
    assert "square" in content


@pytest.mark.asyncio
async def test_deny_with_reason_async() -> None:
    agent = mk_agent(
        tool_policy=lambda tool, agent: "budget exceeded",
        agent_cls=CountingAsyncAgent,
    )
    result = await agent.handle_message_async(llm_tool_msg(agent, x=5))
    content = result_content(result)
    assert agent.calls == 0
    assert "square" in content
    assert "budget exceeded" in content


def test_deny_unrecognized_decision_fails_closed() -> None:
    """A decision of an unrecognized type must NOT be treated as allow."""
    agent = mk_agent(tool_policy=lambda tool, agent: 42)
    result = agent.handle_message(llm_tool_msg(agent, x=5))
    content = result_content(result)
    assert agent.calls == 0
    assert "square" in content


# ------------- contract 4: hook sees parsed tool + bounded context -------------


def test_policy_receives_parsed_tool_and_context() -> None:
    seen: List[Tuple[ToolMessage, ChatAgent, Optional[ChatDocument]]] = []

    def policy(
        tool: ToolMessage,
        agent: ChatAgent,
        chat_doc: Optional[ChatDocument] = None,
    ) -> bool:
        seen.append((tool, agent, chat_doc))
        return True

    agent = mk_agent(tool_policy=policy)
    msg = llm_tool_msg(agent, x=7)
    agent.handle_message(msg)
    assert len(seen) == 1
    tool, seen_agent, chat_doc = seen[0]
    assert isinstance(tool, SquareTool)
    assert tool.request == "square"
    assert tool.x == 7  # final parsed payload
    assert seen_agent is agent
    assert chat_doc is not None
    assert chat_doc.content == msg.content
    assert agent.calls == 1


def test_policy_without_chat_doc_param() -> None:
    """A 2-arg policy (tool, agent) is supported."""
    seen: List[SquareTool] = []

    def policy(tool: ToolMessage, agent: ChatAgent) -> bool:
        assert isinstance(tool, SquareTool)
        seen.append(tool)
        return True

    agent = mk_agent(tool_policy=policy)
    agent.handle_message(llm_tool_msg(agent, x=7))
    assert len(seen) == 1 and seen[0].x == 7
    assert agent.calls == 1


# ---------- contract 5: hook failure => fail-closed, no argument leak ----------


SECRET = "31337"


def failing_policy(tool: ToolMessage, agent: ChatAgent) -> bool:
    # a sloppy policy that embeds tool arguments in its exception
    assert isinstance(tool, SquareTool)
    raise ValueError(f"policy blew up on x={tool.x}")


def test_policy_exception_fails_closed_sync() -> None:
    agent = mk_agent(tool_policy=failing_policy)
    result = agent.handle_message(llm_tool_msg(agent, x=int(SECRET)))
    content = result_content(result)
    assert agent.calls == 0  # handler never ran
    assert "square" in content  # tool name present
    assert SECRET not in content  # raw tool args must not leak
    assert "policy blew up" not in content  # nor the exception message


@pytest.mark.asyncio
async def test_policy_exception_fails_closed_async() -> None:
    agent = mk_agent(
        tool_policy=failing_policy,
        agent_cls=CountingAsyncAgent,
    )
    result = await agent.handle_message_async(llm_tool_msg(agent, x=int(SECRET)))
    content = result_content(result)
    assert agent.calls == 0
    assert "square" in content
    assert SECRET not in content


# -------------------- sync/async callback x dispatch matrix --------------------


@pytest.mark.parametrize("decision", [True, "denied by async policy"])
def test_async_policy_sync_dispatch(decision: Union[bool, str]) -> None:
    async def policy(tool: ToolMessage, agent: ChatAgent) -> Union[bool, str]:
        return decision

    agent = mk_agent(tool_policy=policy)
    result = agent.handle_message(llm_tool_msg(agent, x=4))
    content = result_content(result)
    if decision is True:
        assert "SQUARE_RESULT: 16" in content
        assert agent.calls == 1
    else:
        assert agent.calls == 0
        assert "denied by async policy" in content


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [True, "denied by async policy"])
async def test_async_policy_async_dispatch(decision: Union[bool, str]) -> None:
    async def policy(tool: ToolMessage, agent: ChatAgent) -> Union[bool, str]:
        return decision

    agent = mk_agent(tool_policy=policy, agent_cls=CountingAsyncAgent)
    result = await agent.handle_message_async(llm_tool_msg(agent, x=4))
    content = result_content(result)
    if decision is True:
        assert "SQUARE_RESULT: 16" in content
        assert agent.calls == 1
    else:
        assert agent.calls == 0
        assert "denied by async policy" in content


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [True, "denied by async policy"])
async def test_async_policy_async_dispatch_sync_handler(
    decision: Union[bool, str],
) -> None:
    """Async policy + async dispatch falling back to a SYNC handler: the
    policy must be evaluated correctly on this route too."""

    async def policy(tool: ToolMessage, agent: ChatAgent) -> Union[bool, str]:
        return decision

    agent = mk_agent(tool_policy=policy)  # sync handler only
    result = await agent.handle_message_async(llm_tool_msg(agent, x=4))
    content = result_content(result)
    if decision is True:
        assert "SQUARE_RESULT: 16" in content
        assert agent.calls == 1
    else:
        assert agent.calls == 0
        assert "denied by async policy" in content


@pytest.mark.asyncio
async def test_async_policy_loop_bound_state_async_dispatch_sync_handler() -> None:
    """On the async-dispatch-to-sync-handler route, an async policy must be
    awaited on the CALLER's event loop: a policy awaiting a Future created
    on that loop (a loop-bound resource) must evaluate correctly rather
    than spuriously failing closed."""
    loop = asyncio.get_running_loop()
    fut: "asyncio.Future[bool]" = loop.create_future()
    loop.call_later(0.01, fut.set_result, True)

    async def policy(tool: ToolMessage, agent: ChatAgent) -> bool:
        return await fut

    agent = mk_agent(tool_policy=policy)  # sync handler only
    result = await agent.handle_message_async(llm_tool_msg(agent, x=3))
    assert "SQUARE_RESULT: 9" in result_content(result)
    assert agent.calls == 1


@pytest.mark.asyncio
async def test_sync_policy_async_dispatch_sync_handler() -> None:
    """Async dispatch falling back to a sync handler still runs the
    policy exactly once."""
    seen: List[ToolMessage] = []

    def policy(tool: ToolMessage, agent: ChatAgent) -> bool:
        seen.append(tool)
        return True

    agent = mk_agent(tool_policy=policy)  # sync handler only
    result = await agent.handle_message_async(llm_tool_msg(agent, x=3))
    assert "SQUARE_RESULT: 9" in result_content(result)
    assert agent.calls == 1
    assert len(seen) == 1  # policy consulted exactly once


# ------------------- the hook cannot mutate what executes -------------------


def test_policy_cannot_mutate_tool() -> None:
    """The hook sees copies of the tool AND the chat_doc: mutations through
    either must not reach the handler (no argument transformation)."""

    def policy(
        tool: ToolMessage,
        agent: ChatAgent,
        chat_doc: Optional[ChatDocument] = None,
    ) -> bool:
        assert isinstance(tool, SquareTool)
        tool.x = 999  # tamper via the tool param
        assert chat_doc is not None
        for t in chat_doc.tool_messages:  # tamper via the chat_doc route
            if isinstance(t, SquareTool):
                t.x = 999
        return True

    agent = mk_agent(tool_policy=policy)
    msg = llm_tool_msg(agent, x=3)
    result = agent.handle_message(msg)
    assert "SQUARE_RESULT: 9" in result_content(result)  # 3**2, not 999**2
    assert agent.calls == 1
    # the real chat_doc's parsed tool is untouched too
    assert all(t.x == 3 for t in msg.tool_messages if isinstance(t, SquareTool))


# ------------- non-copyable payload => distinct fail-closed rejection -------------


class LockCarrierTool(ToolMessage):
    request: str = "lock_carrier"
    purpose: str = "A tool whose payload holds a non-copyable object"
    payload: Any = None


class LockCarrierAgent(CountingAgent):
    def lock_carrier(self, msg: LockCarrierTool) -> str:
        self.calls += 1
        return "LOCK_HANDLED"


def test_non_copyable_payload_fails_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A payload the pre-policy deep copy cannot copy (here: a thread lock)
    fails CLOSED even under an allow-all policy, with an operator-log
    diagnostic distinct from 'the policy raised', and an LLM message naming
    only the tool and exception class."""
    agent = LockCarrierAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(response_fn=lambda x: x),
            tool_policy=lambda tool, agent: True,  # allow-all
        )
    )
    agent.enable_message(LockCarrierTool)
    tool = LockCarrierTool(payload=threading.Lock())
    msg = agent.create_llm_response(content="calling tool", tool_messages=[tool])
    with caplog.at_level(logging.ERROR, logger="langroid.agent.tool_policy"):
        result = agent.handle_message(msg)
    content = result_content(result)
    assert agent.calls == 0  # handler never ran
    assert "lock_carrier" in content
    assert "could not be copied" in content
    assert "TypeError" in content  # exception class only...
    assert "_thread.lock" not in content  # ...no payload repr in LLM message
    assert "could not be copied for policy evaluation" in caplog.text
    assert "policy hook raised" not in caplog.text  # distinct diagnostic


# --------------- policy consulted only when a handler exists ---------------


def test_no_handler_policy_not_called_sync() -> None:
    seen: List[ToolMessage] = []

    def policy(tool: ToolMessage, agent: ChatAgent) -> bool:
        seen.append(tool)
        return True

    agent = mk_agent(tool_policy=policy)
    agent.enable_message(NoHandlerTool)
    msg = agent.create_llm_response(NoHandlerTool(y=1).to_json())
    assert agent.handle_message(msg) is None
    assert seen == []


@pytest.mark.asyncio
async def test_no_handler_policy_not_called_async() -> None:
    seen: List[ToolMessage] = []

    def policy(tool: ToolMessage, agent: ChatAgent) -> bool:
        seen.append(tool)
        return True

    agent = mk_agent(tool_policy=policy)
    agent.enable_message(NoHandlerTool)
    msg = agent.create_llm_response(NoHandlerTool(y=1).to_json())
    assert await agent.handle_message_async(msg) is None
    assert seen == []


# ---------- policy gates dispatch even through subclass overrides ----------


class OverridingAgent(CountingAgent):
    """Subclass with a GENERIC handle_tool_message override, as user code
    sometimes does (e.g. to log or wrap every tool execution)."""

    def __init__(self, config: ChatAgentConfig):
        super().__init__(config)
        self.override_calls: int = 0

    def handle_tool_message(
        self,
        tool: ToolMessage,
        chat_doc: Optional[ChatDocument] = None,
    ) -> Union[None, str, ChatDocument]:
        self.override_calls += 1
        if isinstance(tool, SquareTool):
            return f"OVERRIDE_RESULT: {tool.x + 1}"
        return super().handle_tool_message(tool, chat_doc=chat_doc)


def test_override_gated_by_policy_sync() -> None:
    agent = mk_agent(
        tool_policy=lambda tool, agent: "not allowed",
        agent_cls=OverridingAgent,
    )
    assert isinstance(agent, OverridingAgent)
    result = agent.handle_message(llm_tool_msg(agent, x=3))
    assert "blocked by tool policy" in result_content(result)
    assert agent.override_calls == 0  # override never ran
    assert agent.calls == 0


@pytest.mark.asyncio
async def test_override_gated_by_policy_async() -> None:
    agent = mk_agent(
        tool_policy=lambda tool, agent: "not allowed",
        agent_cls=OverridingAgent,
    )
    assert isinstance(agent, OverridingAgent)
    result = await agent.handle_message_async(llm_tool_msg(agent, x=3))
    assert "blocked by tool policy" in result_content(result)
    assert agent.override_calls == 0
    assert agent.calls == 0


def test_override_allowed_by_policy_sync() -> None:
    agent = mk_agent(
        tool_policy=lambda tool, agent: True,
        agent_cls=OverridingAgent,
    )
    assert isinstance(agent, OverridingAgent)
    result = agent.handle_message(llm_tool_msg(agent, x=3))
    assert "OVERRIDE_RESULT: 4" in result_content(result)  # override honored
    assert agent.override_calls == 1


@pytest.mark.asyncio
async def test_override_allowed_by_policy_async() -> None:
    agent = mk_agent(
        tool_policy=lambda tool, agent: True,
        agent_cls=OverridingAgent,
    )
    assert isinstance(agent, OverridingAgent)
    result = await agent.handle_message_async(llm_tool_msg(agent, x=3))
    assert "OVERRIDE_RESULT: 4" in result_content(result)
    assert agent.override_calls == 1


def test_override_no_policy_unchanged_sync() -> None:
    agent = mk_agent(agent_cls=OverridingAgent)
    assert isinstance(agent, OverridingAgent)
    result = agent.handle_message(llm_tool_msg(agent, x=3))
    assert "OVERRIDE_RESULT: 4" in result_content(result)
    assert agent.override_calls == 1


@pytest.mark.asyncio
async def test_override_no_policy_unchanged_async() -> None:
    agent = mk_agent(agent_cls=OverridingAgent)
    assert isinstance(agent, OverridingAgent)
    result = await agent.handle_message_async(llm_tool_msg(agent, x=3))
    assert "OVERRIDE_RESULT: 4" in result_content(result)
    assert agent.override_calls == 1


# ------------------- policy shared by reference across clones -------------------


class BudgetPolicy:
    """Stateful one-use budget: only the first tool call (across ALL agents
    sharing this policy object) is allowed."""

    def __init__(self) -> None:
        self.used: int = 0

    def __call__(self, tool: ToolMessage, agent: ChatAgent) -> bool:
        self.used += 1
        return self.used <= 1


def test_policy_state_shared_across_clones() -> None:
    budget = BudgetPolicy()
    agent = mk_agent(tool_policy=budget)
    clone = agent.clone(1)
    assert isinstance(clone, CountingAgent)
    assert clone.config.tool_policy is budget  # shared by reference

    r1 = agent.handle_message(llm_tool_msg(agent, x=2))
    assert "SQUARE_RESULT: 4" in result_content(r1)
    assert agent.calls == 1

    # the clone consults the SAME budget, which is now exhausted
    r2 = clone.handle_message(llm_tool_msg(clone, x=3))
    assert "blocked by tool policy" in result_content(r2)
    assert clone.calls == 0
    assert budget.used == 2


def test_policy_shared_across_task_wrapped_clone() -> None:
    """Task construction deep-copies the agent's config; the policy must
    stay shared by reference so budget state is enforced globally."""
    budget = BudgetPolicy()
    agent = mk_agent(tool_policy=budget)
    clone = agent.clone(1)
    assert isinstance(clone, CountingAgent)
    Task(clone, interactive=False)  # triggers the config copy in Task.__init__
    assert clone.config.tool_policy is budget  # still shared, not forked

    r1 = agent.handle_message(llm_tool_msg(agent, x=2))
    assert "SQUARE_RESULT: 4" in result_content(r1)
    r2 = clone.handle_message(llm_tool_msg(clone, x=3))
    assert "blocked by tool policy" in result_content(r2)
    assert clone.calls == 0
    assert budget.used == 2


_POLICY_SETS: List[Any] = []


class SetRecordingConfig(ChatAgentConfig):
    """Records every post-construction assignment to `tool_policy`."""

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "tool_policy":
            _POLICY_SETS.append(value)
        super().__setattr__(name, value)


def test_clone_never_mutates_live_policy() -> None:
    """Structural (not timing) check: clone() must never mutate the live
    config's tool_policy -- in particular never set it to None mid-copy,
    which a concurrent dispatch could otherwise observe."""
    budget = BudgetPolicy()
    cfg = SetRecordingConfig(
        name="TestAgent",
        llm=MockLMConfig(response_fn=lambda x: x),
        tool_policy=budget,
    )
    agent = CountingAgent(cfg)
    agent.enable_message(SquareTool)
    _POLICY_SETS.clear()
    before = agent.config.tool_policy
    clone = agent.clone(1)
    assert agent.config.tool_policy is before
    assert before is budget
    assert _POLICY_SETS == []  # no code path assigned tool_policy on the live cfg
    assert clone.config.tool_policy is budget


class LockingPolicy:
    """Policy object holding an unpicklable resource (a thread lock)."""

    def __init__(self) -> None:
        self.lock: threading.Lock = threading.Lock()

    def __call__(self, tool: ToolMessage, agent: ChatAgent) -> bool:
        with self.lock:
            return True

    def __deepcopy__(self, memo: Any) -> "LockingPolicy":
        raise TypeError("cannot deepcopy LockingPolicy (holds a lock)")


def test_clone_with_unpicklable_policy() -> None:
    policy = LockingPolicy()
    agent = mk_agent(tool_policy=policy)
    clone = agent.clone(1)  # must not raise
    assert clone.config.tool_policy is policy
    result = clone.handle_message(llm_tool_msg(clone, x=2))
    assert "SQUARE_RESULT: 4" in result_content(result)


# --------- strict-recovery AnyTool shim: exempt; inner tool checked once ---------


def _strict_recovery_wrapper(agent: CountingAgent, x: int) -> ToolMessage:
    """Enable strict recovery's AnyTool on `agent` (as the recovery code
    does) and return a wrapper instance carrying a SquareTool call."""
    any_tool_class = agent._get_any_tool_message()
    assert any_tool_class is not None
    agent.set_output_format(
        any_tool_class,
        force_tools=True,
        use=True,
        handle=True,
        instructions=True,
    )
    assert agent.output_format is not None  # strict mode is on
    # the dynamic AnyTool class defaults `request`/`purpose`, which mypy
    # cannot see through type[ToolMessage]
    return any_tool_class(tool=SquareTool(x=x))  # type: ignore[call-arg]


def test_strict_recovery_policy_consulted_once() -> None:
    """The AnyTool wrapper is a parsing shim, not a tool execution: a
    one-use budget must be consumed only by the INNER re-parsed tool."""
    budget = BudgetPolicy()
    agent = mk_agent(tool_policy=budget)
    wrapper = _strict_recovery_wrapper(agent, x=5)
    result = agent.handle_message(agent.create_llm_response(wrapper.to_json()))
    assert "SQUARE_RESULT: 25" in result_content(result)
    assert agent.calls == 1
    assert budget.used == 1  # consulted exactly once, on the real tool
    assert agent.output_format is None  # recovery reset strict mode


@pytest.mark.asyncio
async def test_strict_recovery_policy_consulted_once_async() -> None:
    budget = BudgetPolicy()
    agent = mk_agent(tool_policy=budget)
    wrapper = _strict_recovery_wrapper(agent, x=5)
    result = await agent.handle_message_async(
        agent.create_llm_response(wrapper.to_json())
    )
    assert "SQUARE_RESULT: 25" in result_content(result)
    assert agent.calls == 1
    assert budget.used == 1
    assert agent.output_format is None


def test_strict_recovery_denied_inner_tool() -> None:
    """A deny-all policy blocks the INNER tool (normal rejection), but the
    shim itself must still run so strict mode is reset."""
    agent = mk_agent(tool_policy=lambda tool, agent: "no tools allowed")
    wrapper = _strict_recovery_wrapper(agent, x=5)
    result = agent.handle_message(agent.create_llm_response(wrapper.to_json()))
    content = result_content(result)
    assert agent.calls == 0
    assert "blocked by tool policy" in content
    assert "square" in content
    assert agent.output_format is None  # set_output_format(None) still ran


def _chat_doc_aware_policy(
    seen_docs: List[Optional[ChatDocument]],
) -> Any:
    """A policy whose decision depends on message context: it denies when
    no chat_doc is provided, allows when the chat_doc is present."""

    def policy(
        tool: ToolMessage,
        agent: ChatAgent,
        chat_doc: Optional[ChatDocument] = None,
    ) -> Union[bool, str]:
        seen_docs.append(chat_doc)
        if chat_doc is None:
            return "no message context provided"
        return True

    return policy


def test_strict_recovery_policy_receives_chat_doc() -> None:
    """The recovered inner tool's policy check must see the same chat_doc
    a normal dispatch would, not None."""
    seen_docs: List[Optional[ChatDocument]] = []
    agent = mk_agent(tool_policy=_chat_doc_aware_policy(seen_docs))
    wrapper = _strict_recovery_wrapper(agent, x=5)
    msg = agent.create_llm_response(wrapper.to_json())
    result = agent.handle_message(msg)
    assert "SQUARE_RESULT: 25" in result_content(result)
    assert agent.calls == 1
    # the policy saw the real message context (as a copy), not None
    assert len(seen_docs) == 1
    assert seen_docs[0] is not None
    assert seen_docs[0].content == msg.content


@pytest.mark.asyncio
async def test_strict_recovery_policy_receives_chat_doc_async() -> None:
    seen_docs: List[Optional[ChatDocument]] = []
    agent = mk_agent(tool_policy=_chat_doc_aware_policy(seen_docs))
    wrapper = _strict_recovery_wrapper(agent, x=5)
    msg = agent.create_llm_response(wrapper.to_json())
    result = await agent.handle_message_async(msg)
    assert "SQUARE_RESULT: 25" in result_content(result)
    assert agent.calls == 1
    assert len(seen_docs) == 1
    assert seen_docs[0] is not None
    assert seen_docs[0].content == msg.content


def test_injected_exemption_marker_does_not_bypass_policy() -> None:
    """ToolMessage allows extra fields, so an LLM could emit
    `"_tool_policy_exempt": true` inside its tool JSON, planting the marker
    as an INSTANCE attribute. The exemption must resolve on the CLASS only,
    so this spoof must not bypass a deny-all policy."""
    agent = mk_agent(tool_policy=lambda tool, agent: "denied")
    spoofed_json = (
        SquareTool(x=3)
        .to_json()
        .replace(
            '"request":',
            '"_tool_policy_exempt": true, "request":',
            1,
        )
    )
    msg = agent.create_llm_response(spoofed_json)
    parsed = agent.try_get_tool_messages(msg, all_tools=True)
    # the spoofed marker really lands on the parsed instance...
    assert any(getattr(t, "_tool_policy_exempt", False) is True for t in parsed)
    result = agent.handle_message(msg)
    content = result_content(result)
    assert agent.calls == 0  # handler never ran
    assert "blocked by tool policy" in content


def test_strict_recovery_no_policy_unchanged() -> None:
    agent = mk_agent()
    wrapper = _strict_recovery_wrapper(agent, x=5)
    result = agent.handle_message(agent.create_llm_response(wrapper.to_json()))
    assert "SQUARE_RESULT: 25" in result_content(result)
    assert agent.calls == 1
    assert agent.output_format is None


# ---------------- contract 6: composes with USER-origin filter ----------------


def test_policy_does_not_weaken_user_origin_filter() -> None:
    """A handle-only tool arriving as raw USER input must stay vetoed by
    `_filter_user_origin_tools`; the policy hook never even sees it, and
    an allow-everything policy cannot resurrect it."""
    policy_calls: List[ToolMessage] = []

    def policy(tool: ToolMessage, agent: ChatAgent) -> bool:
        policy_calls.append(tool)
        return True

    agent = CountingAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(response_fn=lambda x: x),
            tool_policy=policy,
        )
    )
    # handle-only: LLM cannot use it, so raw USER input must not trigger it
    agent.enable_message(SquareTool, use=False, handle=True)
    user_msg = agent.create_user_response(SquareTool(x=3).to_json())
    assert user_msg.metadata.sender == Entity.USER
    result = agent.handle_message(user_msg)
    assert agent.calls == 0
    assert policy_calls == []
    assert result is None or "SQUARE_RESULT" not in result_content(result)


# ------------------- end-to-end: LLM sees the rejection -------------------


def test_rejection_is_seen_by_llm_in_task_loop() -> None:
    llm_inputs: List[str] = []

    def mock_llm(x: str) -> str:
        llm_inputs.append(x)
        if len(llm_inputs) == 1:
            # the tool argument IS the secret sentinel, so a leak of the
            # raw argument into the rejection would fail the assertion below
            return SquareTool(x=int(SECRET)).to_json()
        return "understood, giving up"

    agent = CountingAgent(
        ChatAgentConfig(
            name="TestAgent",
            llm=MockLMConfig(response_fn=mock_llm),
            tool_policy=lambda tool, agent: "quota exhausted",
        )
    )
    agent.enable_message(SquareTool)
    task = Task(agent, interactive=False)
    task.run("go", turns=4)
    assert agent.calls == 0
    # the LLM's second input is the policy rejection
    assert any("quota exhausted" in s for s in llm_inputs[1:])
    assert all(SECRET not in s for s in llm_inputs)
