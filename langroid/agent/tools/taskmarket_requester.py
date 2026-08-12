"""Taskmarket requester tools with explicit human payment authorization."""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar, Type
from urllib.parse import urlsplit

from pydantic import Field

from langroid.agent.tool_message import ToolMessage

BASE_CHAIN_ID = 8453
BASE_USDC_CONTRACT = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
DEFAULT_TASKMARKET_API_URL = "https://api.taskmarket.dev"
TASKMARKET_WEB_URL = "https://taskmarket.dev"
TASK_ID_PATTERN = re.compile(r"^0x[0-9a-fA-F]{64}$")
WALLET_PATTERN = re.compile(r"^0x[0-9a-fA-F]{40}$")
LEGAL_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
USDC_INPUT_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]{1,6})?$")
TASK_DESCRIPTION_MAX_LENGTH = 10_000
USDC_SCALE = Decimal("1000000")
CLI_RECOVERY_STRING_FIELDS = (
    "idempotencyKey",
    "reason",
    "intentId",
    "intentStatus",
    "operation",
    "txHash",
)

TaskmarketRunner = Callable[[Sequence[str]], Any]
AuthorizationCallback = Callable[[Mapping[str, Any], str], str]
Clock = Callable[[], datetime]


def _safe_cli_recovery_fields(value: Any) -> dict[str, Any]:
    """Return only bounded, non-sensitive Taskmarket recovery metadata."""
    if not isinstance(value, Mapping):
        return {}
    recovery: dict[str, Any] = {}
    status = value.get("status")
    if (
        isinstance(status, int)
        and not isinstance(status, bool)
        and 100 <= status <= 599
    ):
        recovery["status"] = status
    pending = value.get("pending")
    if isinstance(pending, bool):
        recovery["pending"] = pending
    for field in CLI_RECOVERY_STRING_FIELDS:
        item = value.get(field)
        if isinstance(item, str) and 0 < len(item) <= 512 and item.isprintable():
            recovery[field] = item
    return recovery


class TaskmarketRequesterError(RuntimeError):
    """Base error for safe, operator-facing Taskmarket failures."""

    retry_safe = True

    def __init__(
        self, message: str, *, recovery: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.recovery = _safe_cli_recovery_fields(recovery)


class TaskmarketCreationUncertain(TaskmarketRequesterError):
    """Raised when creation may have settled and must not be retried."""

    retry_safe = False


class TaskmarketCommandError(TaskmarketRequesterError):
    """Raised when the first-party Taskmarket CLI fails."""


@dataclass(frozen=True)
class TaskmarketTaskPreview:
    """Immutable description of the exact task creation request."""

    preview_id: str
    description: str
    reward_usdc: str
    duration_hours: int
    deadline: str
    estimated_deadline_utc: str
    deliverables: tuple[str, ...]
    max_spend_usdc: str
    tags: tuple[str, ...]
    preview_expires_at: str

    def as_dict(self) -> dict[str, Any]:
        """Return the complete operator-visible preview."""
        return {
            "preview_id": self.preview_id,
            "mode": "bounty",
            "description": self.description,
            "reward_usdc": self.reward_usdc,
            "duration_hours": self.duration_hours,
            "deadline": self.deadline,
            "estimated_deadline_utc": self.estimated_deadline_utc,
            "deliverables": list(self.deliverables),
            "network": "Base Mainnet",
            "chain_id": BASE_CHAIN_ID,
            "currency": "USDC",
            "usdc_contract": BASE_USDC_CONTRACT,
            "maximum_spend_usdc": self.max_spend_usdc,
            "task_visibility": "public",
            "submission_visibility": "public",
            "tags": list(self.tags),
            "preview_expires_at": self.preview_expires_at,
        }


@dataclass
class _StoredPreview:
    preview: TaskmarketTaskPreview
    expires_at: datetime
    consumed: bool = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_usdc(value: str, field_name: str) -> tuple[Decimal, str]:
    if (
        not isinstance(value, str)
        or len(value) > 64
        or not USDC_INPUT_PATTERN.fullmatch(value)
    ):
        raise TaskmarketRequesterError(
            f"{field_name} must be a plain decimal USDC amount with at most "
            "six decimal places"
        )
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise TaskmarketRequesterError(
            f"{field_name} must be a decimal USDC amount"
        ) from exc
    if not amount.is_finite() or amount <= 0:
        raise TaskmarketRequesterError(f"{field_name} must be greater than zero")
    exponent = amount.as_tuple().exponent
    if not isinstance(exponent, int) or exponent < -6:
        raise TaskmarketRequesterError(
            f"{field_name} supports at most six decimal places"
        )
    normalized = format(amount, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return amount, normalized


def _require_mapping(value: Any, command: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskmarketCommandError(
            f"Taskmarket returned an invalid response for {command}"
        )
    return value


class TaskmarketRequester:
    """Bind guarded requester operations to Langroid tool messages.

    The object deliberately exposes no accept or reject operation. A host
    application must provide ``authorize_creation`` to collect fresh human
    authorization. The callback receives the final preview and a one-time
    statement, and must return the statement exactly.

    Args:
        authorize_creation: Host callback that obtains human authorization.
        hard_maximum_spend_usdc: Host-level ceiling that tool input cannot raise.
        cli_executable: First-party Taskmarket CLI executable.
        api_url: Taskmarket API origin bound to CLI calls and task links.
        accepted_legal_bundle_digest: Exact current draft bundle digest that a
            human or organization reviewed and accepted outside the agent. A
            server-side acceptance receipt takes precedence when available.
        preview_ttl_seconds: Lifetime of a no-spend preview.
        command_timeout_seconds: Per-command CLI timeout.
        runner: Optional runner used by tests or embedded environments.
        clock: Optional UTC clock used by tests.
    """

    def __init__(
        self,
        authorize_creation: AuthorizationCallback | None = None,
        *,
        hard_maximum_spend_usdc: str | None = None,
        cli_executable: str = "taskmarket",
        api_url: str | None = None,
        accepted_legal_bundle_digest: str | None = None,
        preview_ttl_seconds: int = 600,
        command_timeout_seconds: float = 30.0,
        runner: TaskmarketRunner | None = None,
        clock: Clock = _utc_now,
    ) -> None:
        if preview_ttl_seconds <= 0:
            raise ValueError("preview_ttl_seconds must be positive")
        if command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        if not cli_executable.strip():
            raise ValueError("cli_executable must not be empty")

        selected_api_url = (
            api_url or os.getenv("TASKMARKET_API_URL") or DEFAULT_TASKMARKET_API_URL
        )
        parsed_url = urlsplit(selected_api_url)
        if (
            parsed_url.scheme != "https"
            or not parsed_url.netloc
            or parsed_url.username
            or parsed_url.password
            or parsed_url.query
            or parsed_url.fragment
            or parsed_url.path not in {"", "/"}
        ):
            raise ValueError("api_url must be an HTTPS origin without credentials")

        self._authorize_creation = authorize_creation
        if hard_maximum_spend_usdc is None:
            self._hard_maximum_spend: Decimal | None = None
            self._hard_maximum_spend_usdc: str | None = None
        else:
            (
                self._hard_maximum_spend,
                self._hard_maximum_spend_usdc,
            ) = _parse_usdc(hard_maximum_spend_usdc, "hard_maximum_spend_usdc")
        self._cli_executable = cli_executable
        self._api_url = selected_api_url.rstrip("/")
        if accepted_legal_bundle_digest is not None and not (
            LEGAL_DIGEST_PATTERN.fullmatch(accepted_legal_bundle_digest)
        ):
            raise ValueError(
                "accepted_legal_bundle_digest must be sha256 followed by "
                "64 hexadecimal characters"
            )
        self._accepted_legal_bundle_digest = accepted_legal_bundle_digest
        self._preview_ttl = timedelta(seconds=preview_ttl_seconds)
        self._command_timeout = command_timeout_seconds
        self._clock = clock
        self._runner = runner or self._run_cli
        self._previews: dict[str, _StoredPreview] = {}
        self._creation_uncertain = False
        self._lock = threading.RLock()

    @property
    def creation_uncertain(self) -> bool:
        """Whether a prior create requires manual settlement reconciliation."""
        with self._lock:
            return self._creation_uncertain

    def get_tools(self) -> list[Type[ToolMessage]]:
        """Return requester-bound Langroid tool message classes."""
        return [
            TaskmarketPreviewTaskTool.create(self),
            TaskmarketCreateTaskTool.create(self),
            TaskmarketTaskStatusTool.create(self),
            TaskmarketTaskSubmissionsTool.create(self),
        ]

    def preview_task(
        self,
        *,
        description: str,
        reward_usdc: str,
        duration_hours: int,
        deliverables: Sequence[str],
        maximum_spend_usdc: str,
        tags: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Create a no-network, no-spend preview for one bounty task."""
        base_description = description.strip()
        if not base_description or len(base_description) > 8000:
            raise TaskmarketRequesterError(
                "description must contain between 1 and 8000 characters"
            )
        if not 1 <= duration_hours <= 8760:
            raise TaskmarketRequesterError("duration_hours must be between 1 and 8760")

        clean_deliverables = tuple(item.strip() for item in deliverables)
        if not clean_deliverables or len(clean_deliverables) > 20:
            raise TaskmarketRequesterError(
                "deliverables must contain between 1 and 20 items"
            )
        if any(not item or len(item) > 500 for item in clean_deliverables):
            raise TaskmarketRequesterError(
                "each deliverable must contain between 1 and 500 characters"
            )

        reward, normalized_reward = _parse_usdc(reward_usdc, "reward_usdc")
        maximum, normalized_maximum = _parse_usdc(
            maximum_spend_usdc, "maximum_spend_usdc"
        )
        if reward > maximum:
            raise TaskmarketRequesterError("reward_usdc exceeds maximum_spend_usdc")
        if self._hard_maximum_spend is not None and maximum > self._hard_maximum_spend:
            raise TaskmarketRequesterError(
                "maximum_spend_usdc exceeds the host-configured hard cap"
            )

        clean_tags = tuple(tag.strip() for tag in tags if tag.strip())
        if len(clean_tags) > 10 or any(len(tag) > 40 for tag in clean_tags):
            raise TaskmarketRequesterError(
                "tags supports at most 10 values of up to 40 characters"
            )

        exact_description = (
            base_description
            + "\n\nDeliverables:\n"
            + "\n".join(f"- {item}" for item in clean_deliverables)
        )
        if len(exact_description) > TASK_DESCRIPTION_MAX_LENGTH:
            raise TaskmarketRequesterError(
                "description and deliverables exceed Taskmarket's 10000-character "
                "task description limit"
            )
        now = self._clock()
        expires_at = now + self._preview_ttl
        estimated_deadline = now + timedelta(hours=duration_hours)
        preview_id = "tm_" + secrets.token_hex(16)
        preview = TaskmarketTaskPreview(
            preview_id=preview_id,
            description=exact_description,
            reward_usdc=normalized_reward,
            duration_hours=duration_hours,
            deadline=(
                f"{duration_hours} hours after Taskmarket accepts creation "
                "(--duration value)"
            ),
            estimated_deadline_utc=_isoformat(estimated_deadline),
            deliverables=clean_deliverables,
            max_spend_usdc=normalized_maximum,
            tags=clean_tags,
            preview_expires_at=_isoformat(expires_at),
        )
        with self._lock:
            self._previews = {
                key: value
                for key, value in self._previews.items()
                if value.expires_at > now and not value.consumed
            }
            self._previews[preview_id] = _StoredPreview(preview, expires_at)
        preview_data = preview.as_dict()
        preview_data["host_maximum_spend_usdc"] = self._hard_maximum_spend_usdc
        return {"ok": True, "preview": preview_data, "payment_attempted": False}

    def create_task(self, preview_id: str) -> dict[str, Any]:
        """Authorize and execute one task creation without automatic retries."""
        with self._lock:
            if self._creation_uncertain:
                raise TaskmarketCreationUncertain(
                    "A previous task creation has unknown settlement status. "
                    "Reconcile it in Taskmarket before constructing a new requester."
                )
            stored = self._previews.get(preview_id)
            if stored is None:
                raise TaskmarketRequesterError("Unknown task preview")
            if stored.consumed:
                raise TaskmarketRequesterError("Task preview has already been consumed")
            if self._clock() >= stored.expires_at:
                raise TaskmarketRequesterError(
                    "Task preview expired; create a fresh preview"
                )
            if self._hard_maximum_spend is None:
                raise TaskmarketRequesterError(
                    "Creation is disabled until the host configures a hard "
                    "maximum_spend_usdc"
                )

            preflight = self._creation_preflight(stored.preview)
            if self._authorize_creation is None:
                raise TaskmarketRequesterError(
                    "Creation is disabled until the host configures a human "
                    "authorization callback"
                )
            phrase = (
                "AUTHORIZE TASKMARKET CREATE "
                f"{secrets.token_hex(8)} FOR {stored.preview.reward_usdc} USDC "
                f"FROM {preflight['acting_wallet']} ON BASE MAINNET"
            )
            authorization_view = dict(stored.preview.as_dict())
            authorization_view.update(preflight)
            authorization_view["host_maximum_spend_usdc"] = (
                self._hard_maximum_spend_usdc
            )
            try:
                supplied_phrase = self._authorize_creation(authorization_view, phrase)
            except Exception as exc:
                raise TaskmarketRequesterError(
                    "Human authorization callback failed"
                ) from exc
            if not isinstance(supplied_phrase, str) or not secrets.compare_digest(
                supplied_phrase.strip(), phrase
            ):
                raise TaskmarketRequesterError("Human authorization was not granted")

            stored.consumed = True
            self._creation_uncertain = True
            command = [
                "task",
                "create",
                "--description",
                stored.preview.description,
                "--reward",
                stored.preview.reward_usdc,
                "--duration",
                str(stored.preview.duration_hours),
                "--mode",
                "bounty",
                "--task-visibility",
                "public",
                "--submission-visibility",
                "public",
            ]
            if stored.preview.tags:
                command.extend(["--tags", ",".join(stored.preview.tags)])

            try:
                creation = _require_mapping(self._runner(command), "task create")
            except Exception as exc:
                recovery = (
                    exc.recovery if isinstance(exc, TaskmarketRequesterError) else None
                )
                raise TaskmarketCreationUncertain(
                    "Task creation may have settled. Do not retry; reconcile "
                    "the wallet and Taskmarket task history first.",
                    recovery=recovery,
                ) from exc
            task_id = str(creation.get("taskId", ""))
            if not TASK_ID_PATTERN.fullmatch(task_id):
                raise TaskmarketCreationUncertain(
                    "Taskmarket did not return a valid task ID. Do not retry; "
                    "reconcile task history first."
                )

            self._creation_uncertain = False
            task_link = f"{TASKMARKET_WEB_URL}/tasks/{task_id}"
            task_api_url = f"{self._api_url}/api/tasks/{task_id}"
            try:
                status = self.task_status(task_id)["task"]
                status_error = None
            except TaskmarketRequesterError:
                status = None
                status_error = "Created task status could not be retrieved yet"
            return {
                "ok": True,
                "created": True,
                "task_id": task_id,
                "task_link": task_link,
                "task_api_url": task_api_url,
                "task": status,
                "status_error": status_error,
                "acting_wallet": preflight["acting_wallet"],
                "network": "Base Mainnet",
                "chain_id": BASE_CHAIN_ID,
                "spend_usdc": stored.preview.reward_usdc,
                "maximum_spend_usdc": stored.preview.max_spend_usdc,
                "host_maximum_spend_usdc": self._hard_maximum_spend_usdc,
                "automatic_payment_retries": 0,
            }

    def task_status(self, task_id: str) -> dict[str, Any]:
        """Retrieve one task's live status through the first-party CLI."""
        self._validate_task_id(task_id)
        task = _require_mapping(self._runner(["task", "get", task_id]), "task get")
        returned_task_id = str(task.get("id", ""))
        if (
            not TASK_ID_PATTERN.fullmatch(returned_task_id)
            or returned_task_id.lower() != task_id.lower()
        ):
            raise TaskmarketCommandError(
                "Taskmarket returned a mismatched task status response"
            )
        return {
            "ok": True,
            "task_id": task_id,
            "task_link": f"{TASKMARKET_WEB_URL}/tasks/{task_id}",
            "task_api_url": f"{self._api_url}/api/tasks/{task_id}",
            "task": dict(task),
        }

    def task_submissions(self, task_id: str) -> dict[str, Any]:
        """Retrieve submissions for human review without making decisions."""
        self._validate_task_id(task_id)
        submissions = self._runner(["task", "submissions", task_id])
        if not isinstance(submissions, list):
            raise TaskmarketCommandError(
                "Taskmarket returned an invalid response for task submissions"
            )
        checked_submissions: list[dict[str, Any]] = []
        for submission in submissions:
            item = _require_mapping(submission, "task submissions")
            returned_task_id = str(item.get("taskId", ""))
            if (
                not TASK_ID_PATTERN.fullmatch(returned_task_id)
                or returned_task_id.lower() != task_id.lower()
            ):
                raise TaskmarketCommandError(
                    "Taskmarket returned a mismatched submission response"
                )
            checked_submissions.append(dict(item))
        return {
            "ok": True,
            "task_id": task_id,
            "submissions": checked_submissions,
            "review_policy": {
                "human_review_required": True,
                "automatic_acceptance": False,
                "automatic_rejection": False,
            },
        }

    def _creation_preflight(self, preview: TaskmarketTaskPreview) -> dict[str, Any]:
        deposit = _require_mapping(self._runner(["deposit"]), "deposit")
        balance = _require_mapping(
            self._runner(["wallet", "balance"]), "wallet balance"
        )
        legal = _require_mapping(self._runner(["legal", "status"]), "legal status")

        if (
            deposit.get("network") not in {"Base", "Base Mainnet"}
            or deposit.get("chainId") != BASE_CHAIN_ID
            or str(deposit.get("currency", "")).upper() != "USDC"
            or str(deposit.get("usdcContract", "")).lower() != BASE_USDC_CONTRACT
        ):
            raise TaskmarketRequesterError(
                "Taskmarket CLI is not configured for Base Mainnet USDC"
            )

        acting_wallet = str(deposit.get("address", ""))
        balance_wallet = str(balance.get("address", ""))
        if (
            not WALLET_PATTERN.fullmatch(acting_wallet)
            or acting_wallet.lower() != balance_wallet.lower()
        ):
            raise TaskmarketRequesterError(
                "Taskmarket wallet identity does not match its balance response"
            )

        try:
            balance_base_units = int(str(balance["balanceBaseUnits"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise TaskmarketCommandError(
                "Taskmarket returned an invalid wallet balance"
            ) from exc
        reward, _ = _parse_usdc(preview.reward_usdc, "reward_usdc")
        required_base_units = int(reward * USDC_SCALE)
        if balance_base_units < required_base_units:
            raise TaskmarketRequesterError(
                "Taskmarket wallet has insufficient USDC for the exact reward"
            )

        enforcement_enabled = legal.get("enforcementEnabled")
        legal_accepted = legal.get("accepted")
        legal_bundle_digest = legal.get("bundleDigest")
        if not isinstance(enforcement_enabled, bool) or not isinstance(
            legal_accepted, bool
        ):
            raise TaskmarketCommandError(
                "Taskmarket returned an invalid legal status response"
            )
        if not isinstance(
            legal_bundle_digest, str
        ) or not LEGAL_DIGEST_PATTERN.fullmatch(legal_bundle_digest):
            raise TaskmarketCommandError(
                "Taskmarket returned an invalid legal bundle digest"
            )
        if enforcement_enabled and not legal_accepted:
            raise TaskmarketRequesterError(
                "Current Taskmarket legal bundle must be reviewed and accepted "
                "outside the agent before creation"
            )
        if legal_accepted:
            legal_acceptance = "server_receipt"
        elif self._accepted_legal_bundle_digest != legal_bundle_digest:
            raise TaskmarketRequesterError(
                "Current unenforced Taskmarket legal bundle must be explicitly "
                "reviewed and accepted outside the agent; configure its exact "
                "accepted_legal_bundle_digest"
            )
        else:
            legal_acceptance = "host_digest"

        return {
            "acting_wallet": acting_wallet,
            "wallet_balance_usdc": str(balance.get("balanceUsdc", "")),
            "legal_bundle_version": legal.get("bundleVersion"),
            "legal_bundle_status": legal.get("status"),
            "legal_bundle_digest": legal_bundle_digest,
            "legal_acceptance": legal_acceptance,
        }

    def _validate_task_id(self, task_id: str) -> None:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise TaskmarketRequesterError(
                "task_id must be a 0x-prefixed 32-byte hex value"
            )

    def _run_cli(self, arguments: Sequence[str]) -> Any:
        command = [self._cli_executable, *arguments]
        environment = os.environ.copy()
        environment["TASKMARKET_API_URL"] = self._api_url
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._command_timeout,
                check=False,
                shell=False,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TaskmarketCommandError(
                "Taskmarket CLI did not complete; no automatic retry was attempted"
            ) from exc

        if len(completed.stdout) + len(completed.stderr) > 1_000_000:
            raise TaskmarketCommandError("Taskmarket CLI output exceeded 1 MB")
        if completed.returncode != 0:
            recovery: Mapping[str, Any] | None = None
            if list(arguments[:2]) == ["task", "create"]:
                try:
                    error_payload = json.loads(completed.stderr)
                except json.JSONDecodeError:
                    error_payload = None
                if isinstance(error_payload, Mapping):
                    recovery = error_payload
            raise TaskmarketCommandError(
                "Taskmarket CLI command failed; sensitive output was withheld",
                recovery=recovery,
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise TaskmarketCommandError(
                "Taskmarket CLI returned invalid JSON"
            ) from exc
        if not isinstance(payload, Mapping) or payload.get("ok") is not True:
            raise TaskmarketCommandError("Taskmarket CLI reported a failure")
        return payload.get("data")


class _BoundTaskmarketTool(ToolMessage):
    _requester: ClassVar[TaskmarketRequester | None] = None

    @classmethod
    def create(cls, requester: TaskmarketRequester) -> Type["_BoundTaskmarketTool"]:
        """Bind this tool class to one configured requester."""

        class ConfiguredTaskmarketTool(cls):  # type: ignore[misc, valid-type]
            _requester: ClassVar[TaskmarketRequester | None] = requester

        ConfiguredTaskmarketTool.__name__ = cls.__name__
        return ConfiguredTaskmarketTool

    def _configured_requester(self) -> TaskmarketRequester:
        if self._requester is None:
            raise TaskmarketRequesterError(
                "Use TaskmarketRequester.get_tools() to configure this tool"
            )
        return self._requester

    def _render(self, operation: Callable[[], dict[str, Any]]) -> str:
        try:
            result = operation()
        except TaskmarketRequesterError as exc:
            result = {
                "ok": False,
                "error": str(exc),
                "retry_safe": exc.retry_safe,
            }
            if exc.recovery:
                result["recovery"] = exc.recovery
        return json.dumps(result, sort_keys=True)


class TaskmarketPreviewTaskTool(_BoundTaskmarketTool):
    """Create an exact, no-spend preview of a Taskmarket bounty."""

    request: str = "taskmarket_preview_task"
    purpose: str = (
        "Preview an exact Taskmarket bounty before any authorization or payment. "
        "Shows description, deliverables, reward, deadline, Base network, and cap."
    )
    description: str = Field(description="Task objective and acceptance criteria")
    reward_usdc: str = Field(description="Exact reward in decimal USDC")
    duration_hours: int = Field(description="Exact Taskmarket duration in hours")
    deliverables: list[str] = Field(description="Concrete required deliverables")
    maximum_spend_usdc: str = Field(
        description="Task spend ceiling, independently bounded by the host hard cap"
    )
    tags: list[str] = Field(default_factory=list, description="Optional task tags")

    def handle(self) -> str:
        return self._render(
            lambda: self._configured_requester().preview_task(
                description=self.description,
                reward_usdc=self.reward_usdc,
                duration_hours=self.duration_hours,
                deliverables=self.deliverables,
                maximum_spend_usdc=self.maximum_spend_usdc,
                tags=self.tags,
            )
        )


class TaskmarketCreateTaskTool(_BoundTaskmarketTool):
    """Create a previewed task after out-of-band human authorization."""

    request: str = "taskmarket_create_task"
    purpose: str = (
        "Request creation of a valid unexpired preview. The host independently "
        "collects fresh human authorization before any payment."
    )
    preview_id: str = Field(description="Preview ID returned by the preview tool")

    def handle(self) -> str:
        return self._render(
            lambda: self._configured_requester().create_task(self.preview_id)
        )


class TaskmarketTaskStatusTool(_BoundTaskmarketTool):
    """Retrieve a Taskmarket task's current status."""

    request: str = "taskmarket_task_status"
    purpose: str = "Retrieve the live status and link for a Taskmarket task ID."
    task_id: str = Field(description="0x-prefixed Taskmarket task ID")

    def handle(self) -> str:
        return self._render(
            lambda: self._configured_requester().task_status(self.task_id)
        )


class TaskmarketTaskSubmissionsTool(_BoundTaskmarketTool):
    """Retrieve Taskmarket submissions for human review only."""

    request: str = "taskmarket_task_submissions"
    purpose: str = (
        "Retrieve submissions for a Taskmarket task and present them for human "
        "review. This tool cannot accept or reject work."
    )
    task_id: str = Field(description="0x-prefixed Taskmarket task ID")

    def handle(self) -> str:
        return self._render(
            lambda: self._configured_requester().task_submissions(self.task_id)
        )
