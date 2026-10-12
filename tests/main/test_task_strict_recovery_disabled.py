"""Tests that `disable_strict` suppresses `Task`'s final strict-decoding step.

`Task.run()` ends with an optional strict-decoding retry: when the task's
result cannot be parsed into the requested `return_type`, the task asks the
LLM once more, under a strict JSON schema, to resubmit. That retry is gated on
`ChatAgent._json_schema_available()`, which returns False when
`agent.disable_strict` is set.

No existing test covered that gate, because `_json_schema_available()` is also
False for any non-`OpenAIGPT` LLM -- so under `MockLM` the `disable_strict`
assignment is inert and the retry is skipped for the wrong reason. That is the
gap left by the `MockLM` conversion in #1192; see the note at the
`disable_strict` assignment in `tests/main/test_task_run_polymorphic.py`, and
issue #494.

The agent here therefore uses a real `OpenAIGPT` subclass, with its one
network entry point (`chat`) answered locally, so `_json_schema_available()`
is genuinely True and no API key or LLM call is needed.
"""

from typing import Dict, List, Optional, Union

import pytest

import langroid as lr
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.base import (
    LLMFunctionSpec,
    LLMMessage,
    LLMResponse,
    OpenAIJsonSchemaSpec,
    OpenAIToolSpec,
    ToolChoiceTypes,
)
from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig

# The prompt `Task` uses for its strict-decoding retry; seeing it means the
# retry ran. Must stay in sync with `Task.run()` in langroid/agent/task.py.
STRICT_RETRY_MARKER = "adhering to the following JSON schema"


class LocalOpenAIGPT(OpenAIGPT):
    """A real `OpenAIGPT` whose `chat()` is answered locally.

    Subclassing `OpenAIGPT` (rather than using `MockLM`) is the point: the
    `isinstance(self.llm, OpenAIGPT)` arm of `_json_schema_available()` must
    hold. Overriding `chat()` keeps every request off the network, so the test
    needs no API key.
    """

    def __init__(self, config: OpenAIGPTConfig, reply: str) -> None:
        super().__init__(config)
        self.reply = reply
        self.prompts: List[str] = []

    def chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 200,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
    ) -> LLMResponse:
        last = messages if isinstance(messages, str) else messages[-1].content
        self.prompts.append(last or "")
        return LLMResponse(message=self.reply, cached=False)

    def strict_retries(self) -> List[str]:
        """Prompts recorded that came from `Task`'s strict-decoding retry."""
        return [p for p in self.prompts if STRICT_RETRY_MARKER in p]


def _agent_with_local_llm(reply: str) -> ChatAgent:
    """A `ChatAgent` on a local `OpenAIGPT` that always answers with `reply`."""
    llm_config = OpenAIGPTConfig(
        # An explicit dummy key: construction must not depend on the
        # environment, since this test runs in the keyless CI gate.
        api_key="test-no-call-is-made",
        chat_model="gpt-4.1-mini",
        supports_json_schema=True,
    )
    agent = ChatAgent(ChatAgentConfig(llm=llm_config))
    agent.llm = LocalOpenAIGPT(llm_config, reply)
    return agent


def test_disable_strict_suppresses_task_strict_recovery() -> None:
    """`disable_strict` must suppress the strict-decoding retry, and only it.

    Covers the gap left by #1192 (see issue #494). The positive control is
    what makes this test non-vacuous: it proves the retry really does fire on
    this agent when `disable_strict` is False, so the negative case is
    attributable to `disable_strict` and not to some unrelated gate.
    """
    # A reply that cannot be parsed into the requested `return_type` (int),
    # which is what puts `Task.run()` on the strict-decoding path at all.
    unparseable = "I am not an integer."

    # --- positive control: strict recovery DOES run when strict is enabled ---
    agent = _agent_with_local_llm(unparseable)
    llm = agent.llm
    assert isinstance(llm, LocalOpenAIGPT)
    assert agent._json_schema_available(), (
        "precondition failed: the strict-decoding step is gated on "
        "_json_schema_available(), so this agent cannot exercise it"
    )

    task = lr.Task(agent, interactive=False, config=lr.TaskConfig(done_sequences=["L"]))
    result = task[int].run("anything")

    assert result is None, "an unparseable reply must not yield a return_type value"
    assert len(llm.strict_retries()) == 1, (
        "expected exactly one strict-decoding retry when disable_strict is "
        f"False; recorded prompts: {llm.prompts}"
    )

    # --- the behavior under test: disable_strict suppresses that retry ---
    agent = _agent_with_local_llm(unparseable)
    llm = agent.llm
    assert isinstance(llm, LocalOpenAIGPT)
    agent.disable_strict = True

    task = lr.Task(agent, interactive=False, config=lr.TaskConfig(done_sequences=["L"]))
    result = task[int].run("anything")

    assert result is None
    # The assertion that matters: the retry did not happen. Checked before the
    # `_json_schema_available()` assertion below so that removing the
    # `disable_strict` gate fails on observed BEHAVIOR, not on a restatement
    # of the implementation.
    assert llm.strict_retries() == [], (
        "disable_strict=True must suppress Task's strict-decoding retry; "
        f"recorded prompts: {llm.prompts}"
    )
    # ...and it did not happen for the documented reason. Note langroid may
    # itself set `disable_strict` on a strict validation error, so this is
    # asserted on a fresh agent's state only after the behavior check above.
    assert not agent._json_schema_available()


class _LocalTool(lr.ToolMessage):
    """A tool that exists so it can be excluded from strict recovery."""

    request: str = "local_tool"
    purpose: str = "Do something locally."
    x: int

    def handle(self) -> str:
        return "handled"


def _agent_with_no_strict_tool() -> ChatAgent:
    """An agent that wants strict recovery but has no tool to recover into.

    `ChatAgent.llm_response()` retries a failed tool call in strict JSON mode by
    wrapping every *usable* tool in an `AnyTool` schema. `AnyTool` is None when
    strict recovery has been disabled for all of them, which is a state the
    agents themselves create (see `disable_strict_tools_set`).
    """
    agent = _agent_with_local_llm("anything")
    agent.enable_message(_LocalTool)
    agent.disable_strict_tools_set = {"local_tool"}
    agent.tool_error = True  # the last tool call failed validation
    return agent


def test_no_usable_tool_suppresses_strict_recovery() -> None:
    """With no tool to recover into, the sync path declines to respond."""
    agent = _agent_with_no_strict_tool()
    llm = agent.llm
    assert isinstance(llm, LocalOpenAIGPT)
    assert agent._get_any_tool_message() is None, (
        "precondition failed: this agent must have no tool for strict recovery"
    )
    assert agent._json_schema_available()

    assert agent.llm_response("hi") is None
    assert llm.prompts == []


@pytest.mark.asyncio
async def test_no_usable_tool_suppresses_strict_recovery_async() -> None:
    """Async twin of the test above: it must not issue a recovery LLM call.

    Without the `AnyTool is None` guard the async path keeps going: it calls
    `set_output_format(None, ...)` (which resets the saved tool settings),
    builds recovery instructions with the tool schema left out, appends them to
    the user message and calls the LLM again -- so a state the sync path treats
    as "cannot answer" costs an extra request carrying unusable instructions.
    """
    agent = _agent_with_no_strict_tool()
    llm = agent.llm
    assert isinstance(llm, LocalOpenAIGPT)
    assert agent._get_any_tool_message() is None

    assert await agent.llm_response_async("hi") is None
    assert llm.prompts == [], (
        "no usable tool means no strict recovery, so no LLM call should be "
        f"made; recorded prompts: {llm.prompts}"
    )
