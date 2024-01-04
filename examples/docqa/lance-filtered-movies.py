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
from rich import print
from rich.prompt import Prompt
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
    ldb_dir = ".lancedb/data/imdb-reviews"
    rmdir(ldb_dir)
    ldb_cfg = LanceDBConfig(
        cloud=False,
        collection_name="chat-lance-imdb",
        storage_path=ldb_dir,
        embedding=embed_cfg,
    )

    print(
        """
        [blue]Welcome to the IMDB Movies chatbot!
        This dataset has around 130,000 movie reviews, with these columns:
        
        movie, genre, runtime, certificate, rating, stars, 
        description, votes, director.
        
        To keep things speedy, we'll restrict the dataset to movies
        of a specific genre that you can choose.
        """
    )
    genre = Prompt.ask(
        "Which of these genres would you like to focus on?",
        default="Crime",
        choices=[
            "Action",
            "Adventure",
            "Biography",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "History",
            "Horror",
            "Music",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Sport",
            "Thriller",
            "War",
            "Western",
        ],
    )
    cfg = DocChatAgentConfig(
        vecdb=ldb_cfg,
    )
    agent = LanceDocChatAgent(cfg)

    # READ IN AND CLEAN THE DATA
    df = pd.read_csv("examples/docqa/data/movies/IMBD.csv")

    def clean_votes(value):
        """Clean the votes column"""
        # Remove commas and convert to integer, if fails return 0
        try:
            return int(value.replace(",", ""))
        except ValueError:
            return 0

    # Clean the 'votes' column
    df["votes"] = df["votes"].fillna("0").apply(clean_votes)

    # Clean the 'rating' column
    df["rating"] = df["rating"].fillna(0.0).astype(float)

    # Replace missing values in all other columns with '??'
    df.fillna("??", inplace=True)
    df["description"].replace("", "unknown", inplace=True)

    # get the rows where 'Crime' is in the genre column
    df = df[df["genre"].str.contains(genre)]

    print(
        f"""
    [blue]There are {df.shape[0]} movies in {genre} genre, hang on while I load them...
    """
    )
    # sample 1000 rows for faster testing
    df = df.sample(1000)

    # INGEST THE DataFrame into the LanceDocChatAgent
    metadata_cols = []
    agent.ingest_dataframe(df, content="description", metadata=metadata_cols)
    df_description = agent.df_description

    # inform user about the df_description, in blue
    print(
        f"""
    [blue]Here's a description of the DataFrame that was ingested:
    {df_description}
    """
    )

    task = LanceRAGTaskCreator.new(agent, interactive=True)

    task.run("Can you help with some questions about these movies?")


if __name__ == "__main__":
    app()
