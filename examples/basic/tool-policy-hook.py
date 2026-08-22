"""
Example of the pre-execution tool policy hook (`AgentConfig.tool_policy`).

A single agent has a `payment` tool. A policy callback (standing in for an
external authorization/policy engine) rejects any payment above $100; the
handler is then NOT run, and the LLM sees a rejection naming the tool and
the reason, so it can react (here, by telling the user).

See docs/notes/tool-policy-hook.md for the full semantics (allow / reject /
fail-closed on policy errors, sync + async support).

Run like this, optionally specifying an LLM:

python3 examples/basic/tool-policy-hook.py

or

python3 examples/basic/tool-policy-hook.py -m ollama/mistral:7b-instruct-v0.2-q8_0
"""

from fire import Fire

import langroid as lr
import langroid.language_models as lm


class PaymentTool(lr.agent.ToolMessage):
    request: str = "payment"
    purpose: str = "To pay <amount> dollars to <payee>."
    amount: float
    payee: str

    def handle(self) -> str:
        # In real life this would call a payment API
        return f"Paid ${self.amount:.2f} to {self.payee}."


def payment_policy(tool: lr.ToolMessage, agent: lr.Agent) -> bool | str:
    """Reject payments above $100; allow everything else."""
    if isinstance(tool, PaymentTool) and tool.amount > 100:
        return "payments above $100 require manager approval"
    return True


def main(model: str = "") -> None:
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="Payer",
            llm=lm.OpenAIGPTConfig(chat_model=model or lm.OpenAIChatModel.GPT4o),
            system_message="""
            You help the user make payments, using the `payment` tool.
            If a payment is blocked by policy, apologize to the user and
            state the reason; do NOT retry the same payment.
            """,
            tool_policy=payment_policy,
            handle_llm_no_tool="user",  # fwd to user when LLM sends non-tool msg
        )
    )
    agent.enable_message(PaymentTool)
    task = lr.Task(agent, interactive=True)
    # Try asking: "pay $50 to Alice" (allowed),
    # then: "pay $500 to Bob" (blocked by policy; LLM sees the reason)
    task.run("Ask me what payment I'd like to make.")


if __name__ == "__main__":
    Fire(main)
