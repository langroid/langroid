from examples.urlqa.doc_chat_agent import DocChatAgent
from llmagent.parsing.urls import get_urls_from_user
from llmagent.utils.logging import setup_colored_logging
from llmagent.parsing.repo_loader import RepoLoader, RepoLoaderConfig
from llmagent.utils import configuration
from llmagent.agent.base import AgentConfig
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.parsing.parser import ParsingConfig
from llmagent.parsing.code_parser import CodeParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.mytypes import Document

import typer
import os
from rich import print
import warnings

app = typer.Typer()

setup_colored_logging()


class CodeChatConfig(AgentConfig):
    gpt4: bool = False
    cache: bool = True
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

    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        chat_model=OpenAIChatModel.GPT3_5_TURBO,
        use_chat_for_completion=True,
    )
    parsing: ParsingConfig = ParsingConfig(
        splitter="para_sentence",
        chunk_size=500,
        chunk_overlap=50,
    )

    code_parsing: CodeParsingConfig = CodeParsingConfig(
        chunk_size=200,
        token_encoding_model="text-embedding-ada-002",
        extensions=["py", "yml", "yaml", "sh", "md", "txt"],
        n_similar_docs=2,
    )

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )

    repo_url: str = "https://github.com/eugeneyan/testing-ml"


def chat(config: CodeChatConfig) -> None:
    configuration.update_global_settings(config, keys=["debug", "stream", "cache"])
    if config.gpt4:
        config.llm.chat_model = OpenAIChatModel.GPT4
    default_urls = [config.repo_url]

    print("[blue]Welcome to the GitHub Repo chatbot!")
    print("[cyan]Enter x or q to quit, or ? for evidence")
    print("[blue]Enter a GitHub URL below (or leave empty for default Repo)")
    urls = get_urls_from_user(n=1) or default_urls
    loader = RepoLoader(urls[0], config=RepoLoaderConfig())
    dct, documents = loader.load(depth=2, lines=100)
    listing = [
        """
        List of ALL files and directories in this project:
        If a file is not in this list, then we can be sure that
        it is not in the repo!
        """
        ] + loader.ls(dct, depth=1)
    listing = Document(
        content="\n".join(listing),
        metadata={"source": "repo_listing"},
    )

    code_docs = [
        doc for doc in documents if doc.metadata["language"] not in ["md", "txt"]
    ] + [listing]

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
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="do no use cache"),
) -> None:
    config = CodeChatConfig(debug=debug, gpt4=gpt4, cache=not nocache)
    chat(config)


if __name__ == "__main__":
    app()
