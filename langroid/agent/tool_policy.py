"""Machinery for the pre-execution tool policy hook.

The public surface is the ``tool_policy`` field on
:class:`langroid.agent.base.AgentConfig`: an optional callable consulted
just before a parsed ``ToolMessage``'s selected handler runs, deciding
whether the handler executes (allow) or an LLM-visible rejection is
returned instead (reject). The functions here evaluate that hook; see
``docs/notes/tool-policy-hook.md`` for the full semantics.
"""

import asyncio
import copy
import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

if TYPE_CHECKING:
    from langroid.agent.base import Agent, AgentConfig
    from langroid.agent.chat_document import ChatDocument
    from langroid.agent.tool_message import ToolMessage

logger = logging.getLogger(__name__)

C = TypeVar("C", bound="AgentConfig")


class _PolicyCopyError(Exception):
    """Internal marker: the tool/chat_doc could not be copied for the policy.

    Distinguishes a payload-copy failure (e.g. a tool field holding a lock
    or client object) from an exception raised by the policy itself, so the
    operator log can say precisely which happened. Both fail closed.
    """


def dispatch_overridden(agent: "Agent", use_async: bool) -> bool:
    """Whether the agent's tool-dispatch method is overridden by a subclass.

    Async dispatch falls back into `handle_tool_message` when a tool has no
    async handler, so for the async path an override of EITHER method counts.

    Args:
        agent: The agent whose dispatch is being examined.
        use_async: True for the async dispatch path.

    Returns:
        True if a subclass overrides the relevant dispatch method(s).
    """
    from langroid.agent.base import Agent

    if (
        use_async
        and type(agent).handle_tool_message_async is not Agent.handle_tool_message_async
    ):
        return True
    return type(agent).handle_tool_message is not Agent.handle_tool_message


def handler_exists(agent: "Agent", tool: "ToolMessage", use_async: bool) -> bool:
    """Whether base dispatch would find a handler for this tool.

    Args:
        agent: The agent that would dispatch the tool.
        tool: The parsed ToolMessage.
        use_async: True for the async dispatch path, which looks for an
            `<name>_async` handler first and falls back to the sync one.

    Returns:
        True if a handler method exists on the agent.
    """
    tool_name = tool.default_value("request")
    if hasattr(tool, "_handler"):
        handler_name = getattr(tool, "_handler", tool_name)
    else:
        handler_name = tool_name
    if use_async and getattr(agent, handler_name + "_async", None) is not None:
        return True
    return getattr(agent, handler_name, None) is not None


def policy_exempt(tool: "ToolMessage") -> bool:
    """Whether this tool is structurally exempt from the `tool_policy` hook.

    Framework-internal parsing shims declare ``_tool_policy_exempt = True``
    on their class (e.g. the strict-recovery ``AnyTool`` wrapper): such a
    shim is not a tool execution itself -- the actual tool it carries is
    re-parsed and dispatched through the policy-checked path, so the policy
    is consulted exactly once per logical tool call.

    Args:
        tool: The parsed ToolMessage about to be dispatched.

    Returns:
        True if the tool's class carries the exemption marker.
    """
    # SECURITY: resolve the marker on the CLASS only, never the instance.
    # ToolMessage has ``extra="allow"``, so LLM-controlled tool JSON such
    # as ``{"_tool_policy_exempt": true, ...}`` lands on the INSTANCE as an
    # extra field; trusting instance attributes would let the LLM spoof the
    # exemption and bypass the policy.
    return getattr(type(tool), "_tool_policy_exempt", False) is True


def deepcopy_config_sharing_policy(config: C) -> C:
    """Deep-copy an AgentConfig, sharing `tool_policy` by REFERENCE.

    A `tool_policy` hook may be stateful (budgets, approval lists) or hold
    unpicklable resources (locks, clients), so wherever the framework
    deep-copies an agent config (agent cloning, Task construction, batch
    processing) the policy must be exempted from the copy: the copy and the
    original consult the SAME policy object. The live `config` is never
    mutated (the exemption is done by pre-seeding the deepcopy memo).

    Args:
        config: The config to copy.

    Returns:
        A deep copy of `config` whose `tool_policy` is the same object as
        `config.tool_policy`.
    """
    memo: dict[int, Any] = {}
    policy = config.tool_policy
    if policy is not None:
        memo[id(policy)] = policy
    config_copy: C = copy.deepcopy(config, memo)
    return config_copy


def _invoke_policy(
    policy: Callable[..., Any],
    tool: "ToolMessage",
    agent: "Agent",
    chat_doc: Optional["ChatDocument"],
) -> Any:
    """Call the `tool_policy` hook with (tool, agent[, chat_doc]).

    The `chat_doc` is passed only if the callable's signature has a
    `chat_doc` parameter, mirroring the tool-handler convention. The hook
    decides allow/reject only, so it receives deep COPIES of the tool and
    the chat_doc: mutating them cannot change what the handler executes.

    Args:
        policy: The configured `tool_policy` callable.
        tool: The final parsed ToolMessage about to be handled.
        agent: The agent about to run the handler.
        chat_doc: The ChatDocument containing the tool call, if available.

    Returns:
        The policy's decision, possibly an awaitable if the policy
        is async.
    """
    try:
        has_chat_doc_param = "chat_doc" in inspect.signature(policy).parameters
    except (ValueError, TypeError):
        has_chat_doc_param = False
    try:
        tool = tool.model_copy(deep=True)
        if has_chat_doc_param and chat_doc is not None:
            chat_doc = copy.deepcopy(chat_doc)
    except Exception as e:
        raise _PolicyCopyError() from e
    if has_chat_doc_param:
        return policy(tool, agent, chat_doc=chat_doc)
    return policy(tool, agent)


def run_awaitable_sync(awaitable: Any) -> Any:
    """Run an awaitable to completion from sync code.

    Uses `asyncio.run()` when no event loop is running in this thread;
    otherwise runs it on a fresh event loop in a helper thread, so sync
    dispatch nested inside async code (e.g. a sync tool handler invoked
    from async code calling back into sync dispatch) still works.

    Args:
        awaitable: The awaitable to run.

    Returns:
        The awaitable's result.
    """

    async def _await() -> Any:
        return await awaitable

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await())
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, _await()).result()


def _decision_msg(tool_name: str, decision: Any) -> Optional[str]:
    """Convert a `tool_policy` decision into an optional rejection message.

    Args:
        tool_name: Name (`request` value) of the tool being checked.
        decision: The value returned by the `tool_policy` hook.

    Returns:
        None if the tool is allowed (decision is True or None), else an
        LLM-visible rejection message naming the tool and the reason.
        Any decision of an unrecognized type is treated as a rejection
        (fail-closed).
    """
    if decision is None or decision is True:
        return None
    if decision is False:
        reason = "(no reason given)"
    elif isinstance(decision, str):
        reason = decision
    else:
        reason = (
            f"policy returned unrecognized decision of type "
            f"{type(decision).__name__}; failing closed"
        )
    return f"Tool `{tool_name}` was NOT executed: blocked by tool policy: {reason}"


def _copy_failure_msg(tool_name: str, e: BaseException | None) -> str:
    """Fail-closed rejection for a payload that could not be copied.

    A tool field carrying a non-copyable object (a lock, a client, ...)
    prevents the pre-policy deep copy; the tool is blocked (fail-closed),
    with an operator-log diagnostic distinct from "the policy raised". The
    LLM-visible message still names only the tool and the exception class.

    Args:
        tool_name: Name (`request` value) of the tool being checked.
        e: The underlying copy exception, if known.

    Returns:
        An LLM-visible rejection message.
    """
    exc_name = type(e).__name__ if e is not None else "Exception"
    logger.error(
        f"tool payload could not be copied for policy evaluation for tool "
        f"`{tool_name}`; blocking the tool (fail-closed): "
        f"{exc_name}: {e}"
    )
    return (
        f"Tool `{tool_name}` was NOT executed: its arguments could not be "
        f"copied for policy evaluation ({exc_name}); failing closed "
        f"(tool blocked)."
    )


def _failure_msg(tool_name: str, e: Exception) -> str:
    """Fail-closed rejection message for a `tool_policy` hook that raised.

    The exception message is deliberately NOT included, since it may
    embed raw tool arguments; only the exception class name is exposed.
    The full exception is logged for the operator.

    Args:
        tool_name: Name (`request` value) of the tool being checked.
        e: The exception raised by the hook.

    Returns:
        An LLM-visible rejection message.
    """
    logger.error(
        f"tool_policy hook raised {type(e).__name__} while checking tool "
        f"`{tool_name}`; blocking the tool (fail-closed): {e}"
    )
    return (
        f"Tool `{tool_name}` was NOT executed: the tool policy hook "
        f"failed with {type(e).__name__}; failing closed (tool blocked)."
    )


def check_tool_policy(
    policy: Optional[Callable[..., Any]],
    tool: "ToolMessage",
    agent: "Agent",
    chat_doc: Optional["ChatDocument"],
) -> Optional[str]:
    """Evaluate the `tool_policy` hook on the sync dispatch path.

    An async policy is run to completion via `run_awaitable_sync` (which
    works whether or not an event loop is already running in this thread);
    any failure while doing so is treated like any other hook failure:
    the tool is blocked (fail-closed).

    Args:
        policy: The configured `tool_policy` callable, or None.
        tool: The final parsed ToolMessage about to be handled.
        agent: The agent about to run the handler.
        chat_doc: The ChatDocument containing the tool call, if available.

    Returns:
        None if no policy is configured or the policy allows the tool,
        else an LLM-visible rejection message.
    """
    if policy is None:
        return None
    tool_name = tool.default_value("request")
    try:
        decision = _invoke_policy(policy, tool, agent, chat_doc)
        if inspect.isawaitable(decision):
            decision = run_awaitable_sync(decision)
    except _PolicyCopyError as ce:
        return _copy_failure_msg(tool_name, ce.__cause__)
    except Exception as e:
        return _failure_msg(tool_name, e)
    return _decision_msg(tool_name, decision)


async def check_tool_policy_async(
    policy: Optional[Callable[..., Any]],
    tool: "ToolMessage",
    agent: "Agent",
    chat_doc: Optional["ChatDocument"],
) -> Optional[str]:
    """Async version of `check_tool_policy`.

    An async policy is awaited on the CALLING event loop, so it may safely
    use loop-bound resources (locks, clients, futures).

    Args:
        policy: The configured `tool_policy` callable, or None.
        tool: The final parsed ToolMessage about to be handled.
        agent: The agent about to run the handler.
        chat_doc: The ChatDocument containing the tool call, if available.

    Returns:
        None if no policy is configured or the policy allows the tool,
        else an LLM-visible rejection message.
    """
    if policy is None:
        return None
    tool_name = tool.default_value("request")
    try:
        decision = _invoke_policy(policy, tool, agent, chat_doc)
        if inspect.isawaitable(decision):
            decision = await decision
    except _PolicyCopyError as ce:
        return _copy_failure_msg(tool_name, ce.__cause__)
    except Exception as e:
        return _failure_msg(tool_name, e)
    return _decision_msg(tool_name, decision)
