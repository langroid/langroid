from llmagent.parsing.urls import get_urls_from_user
from examples.urlqa.config import URLQAConfig
from examples.urlqa.agent import DocChatAgent
from llmagent.mytypes import Document
from llmagent.parsing.url_loader import URLLoader
from llmagent.utils import configuration
from transformers.utils import logging

import os
from rich import print
import warnings
import hydra
from hydra.core.config_store import ConfigStore

logging.set_verbosity(logging.ERROR) # for transformers logging


# Register the config with Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name=URLQAConfig.__name__, node=URLQAConfig)

@hydra.main(version_base=None, config_name=URLQAConfig.__name__)
def main(config: URLQAConfig) -> None:
    configuration.update_global_settings(config, keys=["debug"])

    default_urls = config.urls

    print("[blue]Welcome to the URL chatbot " "[x or q to quit, ? for evidence]")
    print("[blue]Enter some URLs below (or leave empty for default URLs)")
    urls = get_urls_from_user() or default_urls
    loader = URLLoader(urls=urls)
    documents:List[Document] = loader.load()

    agent = DocChatAgent(config)
    agent.ingest_docs(documents)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #embedding_models = OpenAIEmbeddings()

    print(f"[green] I have processed the following {len(urls)} URLs:")
    print("\n".join(urls))

    warnings.filterwarnings(
        "ignore",
        message="Token indices sequence length.*",
        # category=UserWarning,
        module="transformers",
    )

    while True:
        print("\n[blue]Query: ", end="")
        query = input("")
        if query in ["exit", "quit", "q", "x", "bye"]:
            print("[green] Bye, hope this was useful!")
            break
        if query == "?" and len(response) > 0:
            agent.justify_response()
        elif query.startswith(("summar", "?")) and len(response) == 0:
            agent.summarize_docs()
        else:
            agent.respond(query)

if __name__ == "__main__":
    main()
