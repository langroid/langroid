"""
This example lets you ask questions about GitHub-issues for a repo.

LanceRAGTaskCreator.new(agent) takes a LanceDocChatAgent and sets up a
3-agent system with 2 additional agents:
- QueryPlanner that decides a filter, possibly rephrased query, and
  possibly also dataframe-like calculation to answer things like ("highest rated...")

- QueryPlanAnswerCritic: this looks at the QueryPlan and the answer from the RAG agent
  and suggests changes to the QueryPlan if the answer does not look satisfactory

This system combines:
- filtering using LanceDB (sql-like filtering on document fields
- semantic search using LanceDB (vector search on document content)
- Full Text Search using LanceDB (search on document content)
- Pandas-like dataframe calculations (e.g. "highest rated", "most votes", etc.)

Run like this:
    python examples/docqa/lance-rag-gh-issues.py

Optional arguments:
-nc : turn off caching (i.e. don't retrieve cached LLM responses)
-d: debug mode, to show all intermediate results
"""

import pandas as pd
import typer
from rich.prompt import Prompt

from langroid.agent.special.doc_chat_agent import DocChatAgentConfig
from langroid.agent.special.lance_doc_chat_agent import LanceDocChatAgent
from langroid.agent.special.lance_rag.lance_rag_task import LanceRAGTaskCreator
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.parsing.repo_loader import RepoLoader
from langroid.utils.configuration import Settings, set_global
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
    issue_dicts = [iss.model_dump() for iss in issues]
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

    task = LanceRAGTaskCreator.new(agent, interactive=False)

    while True:
        question = Prompt.ask("What do you want to know? [q to quit]")
        if question == "q":
            break
        result = task.run(question)
        print(
            f"""
            Here's your answer:
            {result.content}
            """
        )


if __name__ == "__main__":
    app()
