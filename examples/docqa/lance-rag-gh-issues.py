"""
2-Agent system for flexible querying of documents using LanceDb for combined
semantic (vector) search, sql-filtering and full-text search,
applied to a collection of GitHub issues from any repo.

The issues (descriptions and metadata) are collected into a dataframe and
directly ingested into LanceDocChatAgent.

Run like this:
    python examples/docqa/lance-filtered-gh-issues.py

Optional arguments:
-nc : turn off caching (i.e. don't retrieve cached LLM responses)
-d: debug mode, to show all intermediate results
"""

import typer
import pandas as pd
from rich.prompt import Prompt
from langroid.parsing.repo_loader import RepoLoader
from langroid.agent.special.doc_chat_agent import DocChatAgentConfig
from langroid.agent.special.lance_doc_chat_agent import LanceDocChatAgent
from langroid.agent.special.lance_rag.lance_rag_task import LanceRAGTaskCreator


from langroid.utils.configuration import set_global, Settings
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.utils.system import rmdir
from langroid.vector_store.lancedb import LanceDBConfig

app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    # Global settings: debug, cache
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )

    # Configs
    embed_cfg = OpenAIEmbeddingsConfig()

    # Get hithub issues
    ldb_dir = ".lancedb/data/gh-issues"
    rmdir(ldb_dir)
    ldb_cfg = LanceDBConfig(
        cloud=False,
        collection_name="chat-lance-gh-issues",
        storage_path=ldb_dir,
        embedding=embed_cfg,
    )

    cfg = DocChatAgentConfig(
        vecdb=ldb_cfg,
        add_fields_to_content=["state", "year", "month", "assignee", "size"],
    )
    agent = LanceDocChatAgent(cfg)
    repo = Prompt.ask(
        "Enter a GitHub repo name as owner/repo, e.g. jmorganca/ollama",
        default="jmorganca/ollama",
    )
    n_issues = Prompt.ask("How many issues to load?", default="100")

    # load github issues from a repo
    repo_loader = RepoLoader(repo)
    issues = repo_loader.get_issues(k=int(n_issues))
    issue_dicts = [iss.dict() for iss in issues]
    df = pd.DataFrame(issue_dicts)
    metadata_cols = []
    agent.ingest_dataframe(df, content="text", metadata=metadata_cols)

    df_description = agent.df_description

    # inform user about the df_description, in blue
    print(
        f"""
    [blue]Here's a description of the DataFrame that was ingested:
    {df_description}
    """
    )

    task = LanceRAGTaskCreator.new(agent, interactive=True)

    task.run("Can you help with some questions?")


if __name__ == "__main__":
    app()
