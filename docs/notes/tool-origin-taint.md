# Tool-Origin Taint Tracking

Langroid marks external user input as *tainted* and propagates that mark through
every mechanical derivation of a message, so that untrusted content can never be
"laundered" into a trusted-looking message that triggers handle-only tools.

## Threat model

`enable_message(MyTool, use=False, handle=True)` registers a tool that an LLM (this
agent's or another agent's) may trigger, but that raw user input must not. The
original fix (GHSA-gjgq-w2m6-wr5q) dropped handle-only tools from USER-origin
messages, and PR #1034 added `ChatDocMetaData.tools_from_agent` so legitimate
multi-agent handoffs (where a Task relabels an agent's output to `sender=USER`)
keep working.

Issue #1035 identified the remaining *laundering* hole: orchestration code that
mechanically re-emits or repackages message content can make USER-derived data look
like trusted AGENT/LLM output. Examples:

- `DonePassTool` parses tool JSON out of the passed message and repackages it into
  an `AgentDoneTool`
- `RewindTool` / `RecipientTool` re-emit attacker-controlled `content` as a new
  `sender=LLM` document
- a tool handler echoes tainted content into its string result, which becomes an
  AGENT-sender document
- `handle_llm_no_tool=DONE` repackages `msg.content` and `msg.tool_messages` into
  an `AgentDoneTool`

## Mechanism

Two marks, both set-only (nothing ever clears taint):

- `ChatDocMetaData.tainted: bool` on every `ChatDocument`: True when the document
  is external user input or was mechanically derived from a tainted document.
- `ToolMessage._tainted: bool` (a private attribute on the `ToolMessage` base):
  True when the tool *object* was parsed out of tainted content, or repackages such
  a tool. Private attributes never appear in LLM-facing schemas or serialized
  JSON, and survive `copy.deepcopy` (hence `ChatDocument.deepcopy`).

Taint sources (documents born tainted):

- `ChatDocument.from_str` (raw string input)
- `Agent.to_ChatDocument` of a USER-authored string, `create_user_response`, and
  the interactive-input path `_user_response_final` (USER sender; SYSTEM input is
  operator-trusted and not tainted)
- `Task.init` / `Task.run` USER-string entry points, and a pre-built USER
  `ChatDocument` handed to a root task
- `ChatDocument.from_LLMMessage` of a `Role.USER` history message

Propagation (mechanical derivations that carry the mark):

- `ChatDocument.deepcopy` (used by `PassTool` / `ForwardTool` and `Task.init`)
- `Agent.get_tool_messages` stamps `_tainted` on every tool parsed out of (or
  attached to) a tainted document, so taint rides the tool objects themselves
  through any subsequent repackaging
- `Agent.response_template` (hence `create_agent_response` / `create_llm_response`)
  taints the new document if any tool passed in `tool_messages` is `_tainted`, or
  if the caller passes `tainted=True`
- `Agent.to_ChatDocument`: a handler result (string or arbitrary object) derived
  while handling a tainted document yields a tainted document
- `Agent._agent_response_final`: the AGENT document wrapping string results of
  handling a tainted message is tainted (content-echo protection)
- `DonePassTool` -> `AgentDoneTool` repackage; the `handle_llm_no_tool=DONE`
  fallback repackage; `SendTool` / `AgentSendTool` re-emission of their own fields
- `RecipientTool` / `AddRecipientTool` / `RewindTool` re-emission
- `Task.result` (the USER-relabeled task result carries the pending message's
  taint), and the `TaskTool` sub-task seed document

Enforcement happens at both places a tool can reach a handler:

- `Agent._filter_user_origin_tools`, applied on every `handle_message` path:
  drops any `_tainted` tool that is not LLM-usable, regardless of which document
  carries it; drops handle-only tools from a tainted document even when
  `tools_from_agent` is set and the sender was relabeled to USER; and drops
  handle-only tools from raw USER-sender documents without `tools_from_agent`
  (the original GHSA fix)
- the recursive handler hop in `Agent.to_ChatDocument`: when a tool handler
  *returns* a ToolMessage, it is normally dispatched directly (never passing the
  filter) — a returned tool that is `_tainted` and not LLM-usable is therefore
  refused execution and packaged into the (tainted) response document instead

LLM-*usable* tools are never filtered: an end user is always allowed to invoke a
tool the LLM itself could have invoked.

## What stays trusted

Legitimate flows are unaffected because taint only enters at external-input
sources:

- LLM emissions (`ChatDocument.from_LLMResponse`) are untainted
- tools an agent's code constructs while handling *untainted* input are
  untainted, and documents built from them (e.g. a handler returning
  `AgentDoneTool(tools=[MyResultTool(...)])`) stay untainted; a tool returned by
  a handler or `handle_message_fallback` while processing a *tainted* document
  inherits the taint (stamped in `Agent.to_ChatDocument` before dispatch)
- a handler that returns its own `ChatDocument` is trusted to label it correctly
  (deriving it via `ChatDocument.deepcopy` preserves taint automatically)

## The irreducible LLM-generation boundary

Taint cannot flow *through* an LLM generation, by design. If a prompt-injected
LLM is manipulated into emitting tool JSON in its content, that emission is
indistinguishable from a legitimate text-format tool call: trusting the LLM's
output is precisely the trust boundary that `use=False, handle=True` expresses
("tools triggered by an LLM's output"). Defending against a compromised LLM
requires application-level measures (e.g. constrained decoding, human approval of
sensitive tools), not origin tracking.

Similarly out of scope: agent code that manually copies field values from a
tainted tool into a newly constructed tool has explicitly chosen to trust those
values; taint tracking cannot follow arbitrary Python data flow.

## History

- GHSA-gjgq-w2m6-wr5q: USER-origin filter for handle-only tools
- PR #1034: `tools_from_agent` preserves multi-agent handoffs
- Issue #1035 Step B (PRs #1038, #1065): `tainted` mark, sources, deepcopy /
  DonePassTool / Task-result / RewindTool / RecipientTool propagation
  (GHSA-4fpx-72j9-gwg3, GHSA-2j3c-5vm9-xppx)
- Issue #1035 Step A: `_tainted` on the `ToolMessage` base, parse-time stamping,
  tool-level veto, and threading through `to_ChatDocument`,
  `_agent_response_final`, the `handle_llm_no_tool=DONE` fallback,
  `SendTool` / `AgentSendTool`, `TaskTool`, and `from_LLMMessage`

Tests: `tests/main/test_tool_origin_taint.py`,
`tests/main/test_tool_taint_laundering.py`.
