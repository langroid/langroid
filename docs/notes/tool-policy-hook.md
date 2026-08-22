# Pre-execution Tool Policy Hook

When Langroid agents are embedded in larger systems, operators often need a
central place to decide whether a tool call the LLM just made should actually
run: authorization checks, human-approval gates, DLP scanning, command
policies, budget limits, and so on. The `AgentConfig.tool_policy` hook
provides the smallest possible interception point for this: a single optional
callback, consulted immediately before the selected tool handler executes.
No policy engine, rules, or registries live in Langroid itself — you plug in
whatever external engine you like.

## API

```python
import langroid as lr

class RunCommandTool(lr.ToolMessage):
    request: str = "run_command"
    purpose: str = "To run a shell <command>"
    command: str

def my_policy(tool: lr.ToolMessage, agent: lr.Agent) -> bool | str | None:
    if isinstance(tool, RunCommandTool) and "rm" in tool.command:
        return "destructive commands are not allowed"
    return True  # allow

config = lr.ChatAgentConfig(
    # ... other params ...
    tool_policy=my_policy,
)
```

The callback is invoked as `tool_policy(tool, agent)` — or as
`tool_policy(tool, agent, chat_doc=...)` if the callable's signature has a
`chat_doc` parameter (mirroring the convention for tool handlers). It
receives:

- `tool`: the final parsed `ToolMessage` instance about to be handled — its
  `request` field is the tool name, and its other fields are the fully
  validated arguments
- `agent`: the agent that is about to run the handler
- `chat_doc` (optional): the `ChatDocument` containing the tool call, when
  available

The return value decides what happens:

- `True` or `None` — allow: the selected handler runs exactly once, as usual
- `False` — reject: the handler is NOT run; the LLM receives a rejection
  message naming the tool
- a `str` — reject, with that string included as the reason in the
  LLM-visible rejection message

The hook cannot modify the tool: argument transformation is deliberately out
of scope, and the hook receives deep copies of both the `ToolMessage` and
the `chat_doc`, so any mutation it makes has no effect on what the handler
sees. It only decides allow vs. reject. It is consulted only when a real
handler has been selected: a tool with no handler returns `None` as before,
without the policy being called.

The policy is enforced at the framework's dispatch call sites, ABOVE
`Agent.handle_tool_message` / `handle_tool_message_async` — so it gates
tool execution even when a subclass overrides those methods. Three
consequences of that placement:

- when dispatch IS overridden, the framework cannot know whether the
  override will execute anything, so the policy is consulted even if the
  override ends up returning `None` (the no-handler shortcut applies only
  to un-overridden dispatch)
- direct calls that YOUR code makes to `agent.handle_tool_message(...)`
  are your own seam and are not gated by the hook
- the strict-recovery `AnyTool` wrapper is a parsing shim, not a tool
  execution, so it is exempt from the hook; the recovered actual tool is
  policy-checked exactly once when dispatched

## Failure semantics: fail-closed

If the policy callback itself raises an exception, the tool is treated as
rejected — the handler does not run. This is deliberate: a broken policy
must never silently become a bypassed policy.

The rejection message sent back to the LLM includes only the tool name and
the exception's class name — never the exception message or the tool's
arguments, since either could leak payload contents into the conversation.
The full exception is logged (via the `langroid.agent.tool_policy` logger)
for the operator. Similarly, a decision value of an unrecognized type (anything other
than `True`, `False`, `None`, or `str`) is treated as a rejection rather
than an allow.

## Sync and async

The hook works on both dispatch paths, with both kinds of callables:

- a sync callback works with both `handle_message` (sync dispatch) and
  `handle_message_async` (async dispatch)
- an async callback is awaited on the caller's event loop everywhere on the
  async path — including when the tool only has a sync handler — so it may
  safely use loop-bound resources (locks, clients, futures). On the sync
  path it is run to completion via `asyncio.run()` — or, when an event loop
  is already running in the calling thread (e.g. sync dispatch invoked from
  inside async code), on a fresh event loop in a helper thread. Any failure
  while doing so is handled fail-closed like any other hook error.

On each dispatch, the policy is consulted exactly once per tool call,
including when async dispatch falls back to a tool's sync handler.

## What the hook does NOT change

- With no `tool_policy` configured (the default), behavior is exactly as
  before — the hook adds negligible overhead (a None check per dispatch)
  and no semantic change.
- The USER-origin tool security filter (see
  [Code-Injection Protection](code-injection-protection.md) and
  GHSA-gjgq-w2m6-wr5q) runs first and is unaffected: tools vetoed by that
  filter are never even shown to the policy, and an allow-everything policy
  cannot resurrect them.
- Rejections flow back into the conversation like any other tool-handling
  result (e.g. validation errors), so the LLM sees why the call was refused
  and can adjust.

## Caveats

The policy callback is trusted operator code, not a sandbox:

- The deep copies guard against mutation through the `tool` and `chat_doc`
  arguments; a policy determined to tamper could still reach live state via
  `agent`. Don't do that — the hook's contract is allow/reject only.
- Everywhere the framework deep-copies an agent config —
  `ChatAgent.clone()`, `Task` construction, and batch processing — the
  policy callable is shared by REFERENCE (exempted from the copy via a
  pre-seeded deepcopy memo; the live config is never mutated). A stateful
  policy (a budget counter, an approval list) is therefore consulted
  globally across the original and all copies, and a policy holding
  unpicklable resources (e.g. a lock) does not break copying. If you copy
  a config through some other route (`copy.deepcopy` yourself), that copy
  is not exempted.
- An async policy evaluated from purely-sync dispatch that is itself
  running inside an event loop (a rare nested case) runs on a fresh event
  loop in a helper thread; loop-bound awaitables can fail there, which
  blocks the tool (fail-closed) rather than bypassing it.
- A tool payload carrying a non-copyable value (a lock, a client object)
  prevents the pre-policy deep copy and causes a fail-closed rejection
  even under an allow-all policy, with a distinct operator-log diagnostic
  ("tool payload could not be copied for policy evaluation"). The remedy
  is keeping tool fields to plain data.

## Example

A runnable example is in `examples/basic/tool-policy-hook.py`, showing a
policy that blocks payments above a limit, with the LLM reacting to the
rejection.
