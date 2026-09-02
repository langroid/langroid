"""A basic chatbot using SerpApi's Google organic search results.

Set SERPAPI_API_KEY in the environment, then run:
    python3 examples/basic/chat-search-serpapi.py
"""

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.serpapi_search_tool import SerpApiSearchTool


def main() -> None:
    config = lr.ChatAgentConfig(
        name="SerpApi Seeker",
        llm=lm.OpenAIGPTConfig(chat_model=lm.OpenAIChatModel.GPT4o),
        system_message=(
            "Use the serpapi_search tool when current web information is needed. "
            "Wait for the tool results before answering and cite their links."
        ),
    )
    agent = lr.ChatAgent(config)
    agent.enable_message(SerpApiSearchTool)
    lr.Task(agent, interactive=True).run("How can I help you?")


if __name__ == "__main__":
    main()
