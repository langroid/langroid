from examples.urlqa.doc_chat_agent import DocChatAgent
from llmagent.parsing.urls import get_urls_from_user
from llmagent.utils.logging import setup_colored_logging
from llmagent.parsing.repo_loader import RepoLoader, RepoLoaderConfig
from llmagent.utils import configuration
from llmagent.agent.base import AgentConfig
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.base import LLMConfig
from llmagent.parsing.parser import ParsingConfig
from llmagent.parsing.code_parser import CodeParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig

import typer
import os
from rich import print
import warnings

app = typer.Typer()

setup_colored_logging()


class CodeChatConfig(AgentConfig):
    gpt4: bool = False
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    max_tokens: int = 10000
    vecdb: VectorStoreConfig = QdrantDBConfig(
        type="qdrant",
        collection_name="llmagent-repo",
        storage_path=".qdrant/data/",
        embedding=OpenAIEmbeddingsConfig(
            model_type="openai",
            model_name="text-embedding-ada-002",
            dims=1536,
        ),
    )

    llm: LLMConfig = LLMConfig(type="openai")
    parsing: ParsingConfig = ParsingConfig(
        splitter="para_sentence",
        chunk_size=500,
        chunk_overlap=50,
    )

    code_parsing: CodeParsingConfig = CodeParsingConfig(
        chunk_size=200,
        token_encoding_model="text-embedding-ada-002",
        extensions=["py", "yml", "yaml", "sh", "md", "txt"],
    )

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )

    repo_url: str = "https://github.com/eugeneyan/testing-ml"


def chat(config: CodeChatConfig) -> None:
    configuration.update_global_settings(config, keys=["debug", "stream"])
    if config.gpt4:
        config.llm.chat_model = "gpt-4"
    default_urls = [config.repo_url]

    print("[blue]Welcome to the GitHub Repo chatbot!")
    print("[cyan]Enter x or q to quit, or ? for evidence")
    print("[blue]Enter a GitHub URL below (or leave empty for default Repo)")
    urls = get_urls_from_user(n=1) or default_urls
    loader = RepoLoader(urls[0], config=RepoLoaderConfig())
    dct, documents = loader.load(depth=2, lines=500)

    code_docs = [
        doc for doc in documents if doc.metadata["language"] not in ["md", "txt"]
    ]

    text_docs = [doc for doc in documents if doc.metadata["language"] in ["md", "txt"]]

    agent = DocChatAgent(config)
    agent.config.parsing = config.parsing
    n_text_splits = agent.ingest_docs(text_docs)
    agent.config.parsing = config.code_parsing
    n_code_splits = agent.ingest_docs(code_docs)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(
        f"""
    [green]I have processed {len(documents)} files from the following GitHub Repo into 
    {n_text_splits} text chunks and {n_code_splits} code chunks:
    {urls[0]}
    """.strip()
    )

    warnings.filterwarnings(
        "ignore",
        message="Token indices sequence length.*",
        # category=UserWarning,
        module="transformers",
    )

    agent.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    gpt4: bool = typer.Option(False, "--gpt4", "-4", help="use GPT-4"),
) -> None:
    config = CodeChatConfig(debug=debug, gpt4=gpt4)
    chat(config)


if __name__ == "__main__":
    app()
