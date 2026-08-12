# Taskmarket requester tools

Langroid can create and monitor [Taskmarket](https://taskmarket.dev/) bounty
tasks through Taskmarket's first-party CLI. The integration separates the
agent's proposed task from the human's payment authorization.

## Setup

Install the CLI and initialize or import its encrypted wallet:

```bash
npm install --global @lucid-agents/taskmarket@latest
taskmarket init                 # or: taskmarket wallet import
taskmarket deposit
taskmarket wallet balance
taskmarket legal status
```

Fund the displayed address with Base Mainnet USDC. The identified human or
organization must review the policy links returned by `taskmarket legal
status`; an agent must not accept them. Use the server-side acceptance receipt
when one is available. If the server identifies the bundle as an unenforced
draft that cannot issue a receipt, configure its exact `bundleDigest` only
after the human or organization explicitly accepts that exact bundle:

```python
requester = TaskmarketRequester(
    authorize_creation=authorize,
    hard_maximum_spend_usdc="5",
    accepted_legal_bundle_digest="sha256:<64 hexadecimal characters>",
)
```

The integration compares this value with the live CLI response before each
creation, so a changed policy bundle fails closed.

## Enable the tools

Configure a callback that displays the final payment details and returns the
one-time statement only when a human types it exactly:

```python
import json
from collections.abc import Mapping
from typing import Any

import langroid as lr
from langroid.agent.tools.taskmarket_requester import TaskmarketRequester


def authorize(view: Mapping[str, Any], statement: str) -> str:
    print(json.dumps(view, indent=2))
    print(f"Type this statement to authorize payment:\n{statement}")
    return input("> ")


requester = TaskmarketRequester(
    authorize_creation=authorize,
    hard_maximum_spend_usdc="5",
    accepted_legal_bundle_digest="sha256:<current accepted draft digest>",
)
agent = lr.ChatAgent(lr.ChatAgentConfig(name="Requester"))
agent.enable_message(requester.get_tools())
```

The agent receives four tools:

- `taskmarket_preview_task` creates a local, ten-minute preview and performs no
  network request or payment.
- `taskmarket_create_task` validates the preview, Base chain ID `8453`, the
  official Base USDC contract, wallet balance, and legal status. It then asks
  the host callback for fresh authorization and runs one CLI create command.
- `taskmarket_task_status` retrieves the created task's live status and link.
- `taskmarket_task_submissions` retrieves work for human review. No acceptance
  or rejection tool is exposed.

Taskmarket defines the deadline as `--duration` hours after the server accepts
creation. The preview shows that exact duration rule and an estimated UTC time;
the status response contains the authoritative `expiryTime`.

## Safety behavior

The task description passed to the CLI is byte-for-byte the previewed
description, including its deliverables. The reward is checked against the
task maximum, an independent host-configured hard cap, and the live USDC balance
before authorization. The combined description is checked against Taskmarket's
10,000-character limit. Commands use an argument vector with `shell=False`;
wallet keys remain inside the CLI keystore and are never accepted as tool inputs.
The configured HTTPS API origin is also injected into the CLI subprocess, so
the network preflight, create call, status lookup, and returned API link cannot
silently target different Taskmarket backends.
Status and submission responses must identify the requested task before they are
returned to the agent.

There is no automatic payment retry. Once an authorized create command starts,
the preview is consumed. A timeout, command failure, or malformed success
response latches that requester instance closed because settlement may have
succeeded. Reconcile the Taskmarket task history and wallet before constructing
a new requester; do not simply repeat the payment. When the CLI returns a JSON
failure, the tool preserves only its bounded recovery metadata (such as the
idempotency key, intent ID/status, and pending flag) while withholding the raw
error text and all unrelated fields.

## Reproduce

Preview safely without payment:

```bash
python examples/basic/taskmarket_requester.py
```

The default run is a no-spend demo. Its abbreviated log is:

```json
{
  "ok": true,
  "preview": {
    "reward_usdc": "1",
    "network": "Base Mainnet",
    "chain_id": 8453,
    "maximum_spend_usdc": "1",
    "host_maximum_spend_usdc": "1"
  },
  "payment_attempted": false
}
```

Add `--create` to enter the human authorization flow. For an unenforced draft,
also pass the exact previously accepted digest with
`--accepted-legal-bundle-digest`. Run the isolated tests:

```bash
pytest -q tests/main/test_taskmarket_requester.py
```

The tests cover exact previews, spend caps, authorization denial, single-use
previews, Base/USDC checks, balance and legal gates, unknown-settlement latching,
read-only submission review, task ID validation, backend binding, and shell-free
execution.
