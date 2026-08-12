"""Preview a Taskmarket bounty and optionally authorize its creation."""

import argparse
import json
from collections.abc import Mapping
from typing import Any

from langroid.agent.tools.taskmarket_requester import TaskmarketRequester


def terminal_authorization(view: Mapping[str, Any], statement: str) -> str:
    """Display exact payment details and collect a human statement."""
    print("\nFinal Taskmarket authorization details:")
    print(json.dumps(view, indent=2))
    print(f"\nType this exact statement to authorize payment:\n{statement}")
    return input("> ")


def main() -> None:
    """Run a no-spend preview or one explicitly authorized creation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create",
        action="store_true",
        help="continue from the preview to the human authorization prompt",
    )
    parser.add_argument(
        "--accepted-legal-bundle-digest",
        help=(
            "exact draft bundle digest already reviewed and accepted by the "
            "human or organization"
        ),
    )
    args = parser.parse_args()

    requester = TaskmarketRequester(
        authorize_creation=terminal_authorization,
        hard_maximum_spend_usdc="1",
        accepted_legal_bundle_digest=args.accepted_legal_bundle_digest,
    )
    preview_result = requester.preview_task(
        description="Implement a documented parser for the supplied data format.",
        reward_usdc="1",
        duration_hours=24,
        deliverables=["Source patch", "Automated test log"],
        maximum_spend_usdc="1",
        tags=["python", "parser"],
    )
    print(json.dumps(preview_result, indent=2))

    if not args.create:
        print("\nPreview only: no network request or payment was attempted.")
        return

    preview_id = str(preview_result["preview"]["preview_id"])
    creation_result = requester.create_task(preview_id)
    print(json.dumps(creation_result, indent=2))


if __name__ == "__main__":
    main()
