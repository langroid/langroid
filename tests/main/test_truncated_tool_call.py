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
from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig

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
    # the raw call still reaches the LLM as plain text, via the existing
    # "treating as normal non-tool message" fallback
    assert response.message is not None
    assert "refund" in response.message


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
