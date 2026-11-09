"""A basic chatbot that searches the web with Firecrawl.

Set FIRECRAWL_API_KEY in the environment, install the `firecrawl` extra
(`pip install "langroid[firecrawl]"`), then run:
    python3 examples/basic/chat-search-firecrawl.py
"""

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.firecrawl_search_tool import FirecrawlSearchTool


def main() -> None:
    config = lr.ChatAgentConfig(
        name="Seeker",
        llm=lm.OpenAIGPTConfig(chat_model=lm.OpenAIChatModel.GPT4o),
        system_message=(
            "Use the firecrawl_search tool when current web information is "
            "needed. Wait for the tool results before answering and cite "
            "their links."
        ),
    )
    agent = lr.ChatAgent(config)
    agent.enable_message(FirecrawlSearchTool)
    lr.Task(agent, interactive=True).run("How can I help you?")


if __name__ == "__main__":
    main()
