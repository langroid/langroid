"""
Agent-loop to classify the intent of a given text.

Run like this (--model is optional, defaults to GPT4o):

python3 examples/basic/intent-classifier.py --model groq/llama-3.1-8b-instant

Other ways to specify the model:
- gpt-4 (set OPENAI_API_KEY in your env or .env file)
- gpt-4o (ditto, set OPENAI_API_KEY)
- cerebras/llama3.1-70b (set CEREBRAS_API_KEY)

For more ways to use langroid with other LLMs, see:
- local/open LLMs: https://langroid.github.io/langroid/tutorials/local-llm-setup/
- non-OpenAPI LLMs: https://langroid.github.io/langroid/tutorials/non-openai-llms/
"""

from enum import Enum
from typing import List, Tuple

from fire import Fire
from rich.prompt import Prompt

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.orchestration import ResultTool


class Intent(str, Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    QUESTION = "question"
    STATEMENT = "statement"


class IntentTool(lr.ToolMessage):
    request: str = "intent_tool"
    purpose: str = """
        To classify the <intent> of a given text, into one of:
        - greeting
        - farewell
        - question
        - statement
        """

    intent: Intent

    @classmethod
    def examples(cls) -> List[lr.ToolMessage | Tuple[str, lr.ToolMessage]]:
        """Use these as few-shot tool examples"""
        return [
            cls(intent=Intent.GREETING),
            ("I want to classify this as a question", cls(intent=Intent.QUESTION)),
        ]

    def handle(self) -> ResultTool:
        """Handle the tool sent by LLM"""

        # ResultTool ends the task
        return ResultTool(intent=self.intent)

    def handle_message_fallback(self, message: lr.ChatDocument) -> str | None:
        """We end up here if the LLM did not send a Tool, so nudge it"""
        if (
            isinstance(message, lr.ChatDocument)
            and message.metadata.sender == lr.Entity.LLM
        ):
            return """
            You forgot to use the `intent_tool` to classify the intent.
            """


def main(model: str = ""):
    intent_tool_name = IntentTool.default_value("request")
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="Intent",
            llm=lm.OpenAIGPTConfig(chat_model=model or lm.OpenAIChatModel.GPT4o),
            use_functions_api=False,  # suppress OpenAI functions/tools
            use_tools=True,  # use langroid-native tools: works with ANY LLM
            system_message=f"""
            You are an astute INTENT CLASSIFIER: given any piece of text
            from the user, you are able to smartly infer their intent.
            Given such a piece of text, classify its intent into one of the following:
             - greeting
             - farewell
             - question
             - statement
            To present your classification, use the 
            `{intent_tool_name}` tool.
            
            ALWAYS use this tool to respond, do NOT say anything else.
            """,
        )
    )

    agent.enable_message(IntentTool)

    # create a task loop specialized to return an Intent
    task = lr.Task(agent=agent, interactive=False)[Intent]

    while True:
        text = Prompt.ask("Enter a text to classify its intent")
        intent = task.run(
            f"""
         Please classify the intent of this text, present your answer
         using the `{intent_tool_name}` tool:
         ----
         {text}
         ----
         """
        )

        print(f"Intent: {intent}")


if __name__ == "__main__":

    Fire(main)
