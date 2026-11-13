"""A tool call cut off by `max_tokens` must not be executed.

When a provider stops a response partway through (`finish_reason="length"`, or
`"content_filter"`), the last `arguments` string it emitted can stop in the
middle of a value. Langroid's lenient JSON parse repairs such a string by
closing the braces the model left open, which silently turns
`{"amount": 1000` into `{"amount": 1}` and `{"amount":` into `{"amount": ""}`,
so the tool runs with values the model never wrote (langroid#1291).

These tests pin the behaviour on both the non-streaming and the streaming
paths, for both the `tool_calls` and the legacy `function_call` shape, and
include controls so a *complete* call is still executed even when the response
as a whole was truncated. No LLM is called.
"""

from typing import Any, Dict, List, Optional

import pytest

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import (
    TRUNCATED_TOOL_CALL_NOTICE,
    TRUNCATING_FINISH_REASONS,
    OpenAIGPT,
    OpenAIGPTConfig,
)

# Arguments the model meant to write, in full.
COMPLETE_ARGS = '{"order_id": "A-77", "amount": 1000}'

# The same arguments as the provider actually delivered them, cut short.
CUT_NUMBER = '{"order_id": "A-77", "amount": 10'  # value truncated: 1000 -> 10
CUT_STRING = '{"to": "x@y.z", "body": "Hello, your order ha'  # mid-sentence
CUT_KEY = '{"order_id": "A-77", "amount":'  # repair_json invents a value here

TRUNCATED_ARGS = [
    pytest.param(CUT_NUMBER, id="cut-number"),
    pytest.param(CUT_STRING, id="cut-string"),
    pytest.param(CUT_KEY, id="cut-key"),
]


@pytest.fixture
def model() -> OpenAIGPT:
    return OpenAIGPT(
        OpenAIGPTConfig(
            chat_model="gpt-4o", api_key="test-key", stream=False, cache_config=None
        )
    )


def _tool_response(args: Any, finish_reason: Optional[str]) -> Dict[str, Any]:
    """An OpenAI chat-completion response carrying one `tool_calls` entry."""
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "refund", "arguments": args},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _function_response(args: Any, finish_reason: Optional[str]) -> Dict[str, Any]:
    """The same, in the legacy `function_call` shape."""
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": None,
                    "function_call": {"name": "refund", "arguments": args},
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _deltas(args: str) -> List[Dict[str, Any]]:
    """Streaming tool deltas that accumulate to `args`."""
    return [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "refund", "arguments": args},
        }
    ]


@pytest.mark.parametrize("args", TRUNCATED_ARGS)
@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_truncated_tool_call_not_kept(
    model: OpenAIGPT, args: str, finish_reason: str
) -> None:
    """A cut-off `tool_calls` entry is dropped, not repaired and executed."""
    response = model._process_chat_completion_response(
        cached=False, response=_tool_response(args, finish_reason)
    )
    assert response.oai_tool_calls is None
    # the LLM is told what happened, so it can retry with shorter arguments --
    # but the cut-off arguments themselves must NOT reach the content, since
    # the agent scans content for tool calls too (see
    # test_truncated_args_are_not_re_extracted_from_content)
    assert response.message == TRUNCATED_TOOL_CALL_NOTICE
    assert args not in response.message


@pytest.mark.parametrize("args", TRUNCATED_ARGS)
def test_truncated_function_call_not_kept(model: OpenAIGPT, args: str) -> None:
    """Same for the legacy `function_call` shape."""
    response = model._process_chat_completion_response(
        cached=False, response=_function_response(args, "length")
    )
    assert response.function_call is None


@pytest.mark.parametrize("args", [None, "", "   "], ids=["null", "empty", "blank"])
def test_call_cut_before_any_arguments_not_kept(model: OpenAIGPT, args: Any) -> None:
    """A response cut off after the tool *name* but before its arguments.

    There is nothing to JSON-validate here, and such a call used to survive
    with `arguments=None` -- which means the tool then ran on whatever defaults
    its schema supplies, with no value from the model at all.
    """
    tool = model._process_chat_completion_response(
        cached=False, response=_tool_response(args, "length")
    )
    assert tool.oai_tool_calls is None
    fun = model._process_chat_completion_response(
        cached=False, response=_function_response(args, "length")
    )
    assert fun.function_call is None


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls", "length"])
def test_no_arg_call_still_kept(model: OpenAIGPT, finish_reason: str) -> None:
    """Control: a tool that genuinely takes no arguments sends `{}`, and runs.

    Including in a response that *was* truncated: `{}` is complete JSON, so
    nothing was lost.
    """
    response = model._process_chat_completion_response(
        cached=False, response=_tool_response("{}", finish_reason)
    )
    assert response.oai_tool_calls is not None
    assert len(response.oai_tool_calls) == 1


@pytest.mark.parametrize("args", TRUNCATED_ARGS)
def test_truncated_streaming_tool_call_not_kept(model: OpenAIGPT, args: str) -> None:
    """The streaming assembly path drops a cut-off call too."""
    response, cached = model._create_stream_response(
        chat=True, tool_deltas=_deltas(args), finish_reason="length"
    )
    assert response.oai_tool_calls is None
    # the response cached as if it were non-streaming must not carry it either,
    # or a later cache hit would resurrect the truncated call
    assert "tool_calls" not in cached["choices"][0]["message"]


def test_streamed_call_cut_before_any_arguments_not_kept(model: OpenAIGPT) -> None:
    """A stream cut off after the tool *name* but before its arguments.

    There is nothing to JSON-validate here, and the empty-argument call used to
    survive the parse with `arguments=None` -- which means the tool then ran on
    whatever defaults its schema supplies, with no value from the model at all.
    """
    response, cached = model._create_stream_response(
        chat=True, tool_deltas=_deltas(""), finish_reason="length"
    )
    assert response.oai_tool_calls is None
    assert "tool_calls" not in cached["choices"][0]["message"]


def test_streamed_function_call_cut_before_any_arguments_not_kept(
    model: OpenAIGPT,
) -> None:
    """Same, in the legacy `function_call` shape."""
    response, cached = model._create_stream_response(
        chat=True,
        has_function=True,
        function_name="refund",
        function_args="",
        finish_reason="length",
    )
    assert response.function_call is None
    assert "function_call" not in cached["choices"][0]["message"]


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls"])
def test_streamed_no_arg_call_still_kept_when_not_truncated(
    model: OpenAIGPT, finish_reason: str
) -> None:
    """Control: a genuinely argument-less call in a normal response still runs.

    Some providers send `arguments=""` rather than `"{}"` for a no-argument
    tool; that must keep working when the response was not cut short.
    """
    response, _ = model._create_stream_response(
        chat=True, tool_deltas=_deltas(""), finish_reason=finish_reason
    )
    assert response.oai_tool_calls is not None
    assert len(response.oai_tool_calls) == 1
    assert response.oai_tool_calls[0].function.name == "refund"


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls", "length"])
def test_complete_tool_call_still_kept(model: OpenAIGPT, finish_reason: str) -> None:
    """Control: complete arguments are executed, truncated response or not.

    The `length` case matters most: a response can hit the output limit *after*
    the model finished writing its tool call, and that call is still good.
    """
    response = model._process_chat_completion_response(
        cached=False, response=_tool_response(COMPLETE_ARGS, finish_reason)
    )
    assert response.oai_tool_calls is not None
    assert len(response.oai_tool_calls) == 1
    assert response.oai_tool_calls[0].function is not None
    assert response.oai_tool_calls[0].function.arguments == {
        "order_id": "A-77",
        "amount": 1000,
    }


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls"])
def test_imperfect_json_still_repaired_when_not_truncated(
    model: OpenAIGPT, finish_reason: str
) -> None:
    """Control: lenient repair is untouched for a response that ended normally.

    Single-quoted args are not valid JSON but are complete; providers that emit
    them must keep working.
    """
    response = model._process_chat_completion_response(
        cached=False,
        response=_tool_response("{'order_id': 'A-77', 'amount': 1000}", finish_reason),
    )
    assert response.oai_tool_calls is not None
    assert response.oai_tool_calls[0].function.arguments == {
        "order_id": "A-77",
        "amount": 1000,
    }


def test_only_the_truncated_call_is_dropped(model: OpenAIGPT) -> None:
    """A response whose *last* call was cut keeps the complete earlier ones."""
    response = model._process_chat_completion_response(
        cached=False,
        response={
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "refund",
                                    "arguments": '{"order_id": "A-1", "amount": 5}',
                                },
                            },
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {
                                    "name": "email",
                                    "arguments": '{"text": "hel',
                                },
                            },
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    assert response.oai_tool_calls is not None
    assert [t.id for t in response.oai_tool_calls] == ["c1"]


# module-level (not a class attribute: ToolMessage is a pydantic model, where a
# class attribute would become a field) so the test can see whether the handler
# ran at all, and with what
HANDLER_RAN_WITH: List[float] = []


class Refund(ToolMessage):
    request: str = "refund"
    purpose: str = "Refund <amount> dollars on order <order_id>."
    order_id: str
    amount: float

    def handle(self) -> str:
        HANDLER_RAN_WITH.append(self.amount)
        return f"refunded ${self.amount}"


def test_agent_does_not_run_handler_for_truncated_call(model: OpenAIGPT) -> None:
    """End to end: the agent does not execute a tool cut off by `max_tokens`."""
    agent = ChatAgent(
        ChatAgentConfig(
            llm=OpenAIGPTConfig(
                chat_model="gpt-4o", api_key="test-key", stream=False, cache_config=None
            ),
            use_tools=False,
            use_functions_api=True,
            use_tools_api=True,
        )
    )
    agent.enable_message(Refund)

    HANDLER_RAN_WITH.clear()
    truncated = ChatDocument.from_LLMResponse(
        model._process_chat_completion_response(
            cached=False, response=_tool_response(CUT_NUMBER, "length")
        )
    )
    agent.agent_response(truncated)
    assert HANDLER_RAN_WITH == []

    # control: the complete call the model meant to send does run
    HANDLER_RAN_WITH.clear()
    complete = ChatDocument.from_LLMResponse(
        model._process_chat_completion_response(
            cached=False, response=_tool_response(COMPLETE_ARGS, "tool_calls")
        )
    )
    result = agent.agent_response(complete)
    assert HANDLER_RAN_WITH == [1000.0]
    assert result is not None and "1000" in result.content


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_truncated_args_are_not_re_extracted_from_content(
    model: OpenAIGPT, stream: bool
) -> None:
    """Dropping the call is not enough: its text must not reach the content.

    The agent also scans message *content* for tool calls, and a cut-off
    arguments string can carry a complete-looking JSON fragment -- here a
    whole element of a list the model never finished. Echoing the raw
    arguments back as content would let that fragment be extracted and run,
    reinstating the very bug by another door.
    """
    cut = '{"items": [{"order_id": "A-1", "amount": 1000}], "memo": "cut off'
    agent = ChatAgent(
        ChatAgentConfig(
            llm=OpenAIGPTConfig(
                chat_model="gpt-4o", api_key="test-key", stream=False, cache_config=None
            ),
            use_tools=False,
            use_functions_api=True,
            use_tools_api=True,
        )
    )
    agent.enable_message(Refund)

    if stream:
        response, _ = model._create_stream_response(
            chat=True, tool_deltas=_deltas(cut), finish_reason="length"
        )
    else:
        response = model._process_chat_completion_response(
            cached=False, response=_tool_response(cut, "length")
        )
    assert response.oai_tool_calls is None
    assert response.message is not None
    assert "amount" not in response.message

    HANDLER_RAN_WITH.clear()
    agent.agent_response(ChatDocument.from_LLMResponse(response))
    assert HANDLER_RAN_WITH == []


def _reasoning_tool_response(
    args: Any, finish_reason: Optional[str], shape: str
) -> Dict[str, Any]:
    """A response whose content carries inline `<think>` tags AND a cut call."""
    message: Dict[str, Any] = {
        "content": "<think>I should refund the full amount</think>Refunding now.",
    }
    if shape == "tool_calls":
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "refund", "arguments": args},
            }
        ]
    else:
        message["function_call"] = {"name": "refund", "arguments": args}
    return {
        "choices": [{"finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.parametrize("shape", ["tool_calls", "function_call"])
def test_discard_notice_survives_inline_reasoning(model: OpenAIGPT, shape: str) -> None:
    """The notice must reach the LLM's next turn, reasoning tags or not.

    When inline thought tags are extracted, `message_with_reasoning` holds the
    original text and `ChatDocument.to_LLMMessage` PREFERS it over `message`.
    So a notice appended only to `message` is silently dropped from history,
    and the model never learns its call was discarded -- the retry prompt this
    fix exists to deliver never arrives.
    """
    response = model._process_chat_completion_response(
        cached=False,
        response=_reasoning_tool_response(CUT_NUMBER, "length", shape),
    )
    # the call itself is still discarded, and the notice is in the message
    assert response.oai_tool_calls is None
    assert response.function_call is None
    assert TRUNCATED_TOOL_CALL_NOTICE in (response.message or "")

    # ... and it survives the round-trip into the next request's history
    doc = ChatDocument.from_LLMResponse(response)
    llm_messages = ChatDocument.to_LLMMessage(doc)
    assert any(
        TRUNCATED_TOOL_CALL_NOTICE in (m.content or "") for m in llm_messages
    ), f"notice lost from history: {[m.content for m in llm_messages]}"
    # the reasoning-bearing copy is still what preserves the thought tags
    assert "<think>" in (response.message_with_reasoning or "")


def test_discard_notice_survives_inline_reasoning_streaming(model: OpenAIGPT) -> None:
    """The streaming path must hold the same guarantee as the non-streaming one."""
    response, _ = model._create_stream_response(
        chat=True,
        completion="<think>I should refund the full amount</think>Refunding now.",
        content_present=True,
        tool_deltas=_deltas(CUT_NUMBER),
        finish_reason="length",
    )
    assert response.oai_tool_calls is None
    doc = ChatDocument.from_LLMResponse(response)
    assert any(
        TRUNCATED_TOOL_CALL_NOTICE in (m.content or "")
        for m in ChatDocument.to_LLMMessage(doc)
    )
    assert "<think>" in (response.message_with_reasoning or "")


# A COMPLETE arguments string that `json.loads` rejects only because it carries a
# raw newline inside a string value. `LLMFunctionCall.from_dict` strips newlines
# before parsing, precisely because some providers emit them, so this call is
# executed normally when the response ends cleanly -- and must not be discarded
# merely because the response was truncated after it.
COMPLETE_ARGS_WITH_RAW_NEWLINE = (
    '{"order_id": "A-77", "amount": 1000,\n "memo": "line1\nline2"}'  # noqa: E501
)


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_complete_args_with_raw_newline_still_kept(
    model: OpenAIGPT, finish_reason: str
) -> None:
    """A strict gate must not discard a call the model finished writing.

    The guarded parse tolerates raw newlines inside argument strings; a bare
    `json.loads` does not. Gating on the bare parse dropped a complete call
    whenever the response happened to be truncated later -- a working agent
    would start losing calls and retry-looping on providers that emit them.
    """
    response = model._process_chat_completion_response(
        cached=False,
        response=_tool_response(COMPLETE_ARGS_WITH_RAW_NEWLINE, finish_reason),
    )
    assert response.oai_tool_calls is not None, "complete call wrongly discarded"
    assert len(response.oai_tool_calls) == 1
    args = response.oai_tool_calls[0].function.arguments
    assert args is not None and args["amount"] == 1000
    assert TRUNCATED_TOOL_CALL_NOTICE not in (response.message or "")

    # control: the same response ending cleanly keeps it too, identically
    clean = model._process_chat_completion_response(
        cached=False,
        response=_tool_response(COMPLETE_ARGS_WITH_RAW_NEWLINE, "tool_calls"),
    )
    assert clean.oai_tool_calls is not None
    assert clean.oai_tool_calls[0].function.arguments == args

    # and a genuinely cut-off string containing a raw newline is STILL discarded
    cut = '{"order_id": "A-77",\n "memo": "line1\nline2'
    dropped = model._process_chat_completion_response(
        cached=False, response=_tool_response(cut, finish_reason)
    )
    assert dropped.oai_tool_calls is None
    assert TRUNCATED_TOOL_CALL_NOTICE in (dropped.message or "")


def test_notice_names_no_cause_so_it_fits_content_filter(model: OpenAIGPT) -> None:
    """Both truncating finish_reasons reach the notice, so it must fit both.

    `content_filter` is not the output token limit, and the notice is the text
    the model is asked to act on, so it must not assert a cause that is wrong
    half the time.
    """
    assert "token limit" not in TRUNCATED_TOOL_CALL_NOTICE
    for finish_reason in TRUNCATING_FINISH_REASONS:
        response = model._process_chat_completion_response(
            cached=False, response=_tool_response(CUT_NUMBER, finish_reason)
        )
        assert TRUNCATED_TOOL_CALL_NOTICE in (response.message or "")
