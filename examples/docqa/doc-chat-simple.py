"""
Bare-bones example of using DocChatAgent to query a document.

Run like this (omit the model to use default GPT-4o):
    
    python3 examples/docqa/doc-chat-simple.py --model ollama/qwen2.5:latest
    
"""

from fire import Fire

import langroid.language_models as lm
from langroid.agent.special.doc_chat_agent import (
    DocChatAgent,
    DocChatAgentConfig,
)


def main(model: str = ""):
    # set up the agent
    agent = DocChatAgent(
        DocChatAgentConfig(
            llm=lm.OpenAIGPTConfig(chat_model=model or lm.OpenAIChatModel.GPT4o),
            # several configs possible here, omitted for brevity
        )
    )

    # ingest document(s), could be a local file/folder or URL
    # Try Borges' "Library of Babel" short story
    url = "https://xpressenglish.com/our-stories/library-of-babel/"

    agent.ingest_doc_paths([url])

    result = agent.llm_response("what is the shape of the rooms in the library?")

    assert "hexagon" in result.content.lower()

    print(result.content)


if __name__ == "__main__":
    Fire(main)
