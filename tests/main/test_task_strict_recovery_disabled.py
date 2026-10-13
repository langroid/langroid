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


class _RecoveryTool(lr.ToolMessage):
    """A tool that exists so strict recovery has something to recover into."""

    request: str = "recovery_tool"
    purpose: str = "Retry a failed tool call."

    def response(self, agent: ChatAgent) -> str:
        return "recovered"


class _AsyncRecordingLLM(LocalOpenAIGPT):
    """`LocalOpenAIGPT` that also answers `achat`, recording the entry point.

    Strict recovery re-asks the LLM under the `AnyTool` JSON schema. The sync
    twin must reach the LLM through `chat` and the async twin through `achat`;
    recording both makes that observable with no network call.
    """

    def __init__(self, config: OpenAIGPTConfig, reply: str) -> None:
        super().__init__(config, reply)
        self.calls: List[str] = []

    def chat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 200,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str | Dict[str, str]] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
    ) -> LLMResponse:
        self.calls.append("chat")
        return super().chat(
            messages,
            max_tokens,
            tools,
            tool_choice,
            functions,
            function_call,
            response_format,
        )

    async def achat(
        self,
        messages: Union[str, List[LLMMessage]],
        max_tokens: int = 200,
        tools: Optional[List[OpenAIToolSpec]] = None,
        tool_choice: ToolChoiceTypes | Dict[str, str | Dict[str, str]] = "auto",
        functions: Optional[List[LLMFunctionSpec]] = None,
        function_call: str | Dict[str, str | Dict[str, str]] = "auto",
        response_format: Optional[OpenAIJsonSchemaSpec] = None,
    ) -> LLMResponse:
        self.calls.append("achat")
        last = messages if isinstance(messages, str) else messages[-1].content
        self.prompts.append(last or "")
        return LLMResponse(message=self.reply, cached=False)


def _agent_awaiting_strict_recovery(reply: str) -> ChatAgent:
    """An agent whose next LLM call must go through strict recovery.

    `tool_error=True` plus at least one usable tool is exactly the state in
    which `ChatAgent.llm_response[_async]()` re-asks the LLM under the
    `AnyTool` JSON schema (the `strict_recovery` block of both twins).
    """
    llm_config = OpenAIGPTConfig(
        api_key="test-no-call-is-made",
        chat_model="gpt-4.1-mini",
        supports_json_schema=True,
    )
    agent = ChatAgent(ChatAgentConfig(llm=llm_config))
    agent.llm = _AsyncRecordingLLM(llm_config, reply)
    agent.enable_message([_RecoveryTool])
    agent.tool_error = True
    agent.set_output_format(None)
    assert agent._json_schema_available()
    assert agent._get_any_tool_message() is not None
    return agent


def test_strict_recovery_sync_uses_sync_llm_call() -> None:
    """Control: the sync twin's recovery retry reaches the LLM via `chat`."""
    agent = _agent_awaiting_strict_recovery("recovered")
    llm = agent.llm
    assert isinstance(llm, _AsyncRecordingLLM)

    agent.llm_response("hi")

    assert llm.calls == [
        "chat"
    ], f"the sync twin must reach the LLM through `chat`; got {llm.calls}"


@pytest.mark.asyncio
async def test_strict_recovery_async_uses_async_llm_call() -> None:
    """The async twin's recovery retry must reach the LLM via `achat`.

    The `strict_recovery` block of `llm_response_async()` is a twin of the
    sync one, and every other statement in it dispatches to the async
    variant (`llm_response_messages_async`, ...). Before the fix it reused
    the sync twin's recursive `self.llm_response(...)` call, so the retry
    was a blocking sync request issued from inside the coroutine, bypassing
    the async streaming path entirely.
    """
    agent = _agent_awaiting_strict_recovery("recovered")
    llm = agent.llm
    assert isinstance(llm, _AsyncRecordingLLM)

    result = await agent.llm_response_async("hi")

    assert result is not None
    assert llm.calls == ["achat"], (
        "the async twin's strict-recovery retry must reach the LLM through "
        f"`achat`, not the blocking sync `chat`; got {llm.calls}"
    )


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
