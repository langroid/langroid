# How to setup Langroid and Pinecone Serverless
This document serves as a quick tutorial on how to use [Pinecone](https://www.pinecone.io/)
Serverless Indexes with Langroid. We will go over some quickstart links and 
some code snippets on setting up a conversation with an LLM utilizing Langroid.

# Setting up Pinecone
Here are some reference links if you'd like to read a bit more on Pinecone's
model definitions and API:
- https://docs.pinecone.io/guides/get-started/overview
- https://docs.pinecone.io/guides/get-started/glossary
- https://docs.pinecone.io/guides/indexes/manage-indexes
- https://docs.pinecone.io/reference/api/introduction
## Signing up for Pinecone
To get started, you'll need to have an account. [Here's](https://www.pinecone.io/pricing/) where you can review the
pricing options for Pinecone. Once you have an account, you'll need to procure an API
key. Make sure to save the key you are given on initial login in a secure location. If
you were unable to save it when your account was created, you can always [create a new
API key](https://docs.pinecone.io/guides/projects/manage-api-keys) in the pinecone console.
## Setting up your local environment
For the purposes of this example, we will be utilizing OpenAI for the generation of our
embeddings. As such, alongside a Pinecone API key, you'll also want an OpenAI key. You can
find a quickstart guide on getting started with OpenAI (here)[https://platform.openai.com/docs/quickstart].
Once you have your API key handy, you'll need to enrich your `.env` file with it.
You should have something like the following:
```env
...
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
PINECONE_API_KEY=<YOUR_PINECONE_API_KEY>
...
```

# Using Langroid with Pinecone Serverless
Once you have completed signing up for an account and have added your API key
to your local environment, you can start utilizing Langroid with Pinecone.
## Setting up an Agent
Here's some example code setting up an agent:
```python
from langroid import Document, DocMetaData
from langroid.agent.special import DocChatAgent, DocChatAgentConfig
from langroid.embedding_models import OpenAIEmbeddingsConfig
from langroid.language_models import OpenAIGPTConfig, OpenAIChatModel
from langroid.parsing.parser import ParsingConfig, Splitter
from langroid.vector_store import PineconeDBConfig

agent_embed_cfg = OpenAIEmbeddingsConfig(
    model_type="openai"
)

agent_config = DocChatAgentConfig(
    llm=OpenAIGPTConfig(
        chat_model=OpenAIChatModel.GPT4o_MINI
    ),
    vecdb=PineconeDBConfig(
        # note, Pinecone indexes must be alphanumeric lowercase characters or "-"
        collection_name="pinecone-serverless-example",
        replace_collection=True,
        embedding=agent_embed_cfg,
    ),
    parsing=ParsingConfig(
        separators=["\n"],
        splitter=Splitter.SIMPLE,
    ),
    n_similar_chunks=2,
    n_relevant_chunks=2,
)

agent = DocChatAgent(config=agent_config)

###################
# Once we have created an agent, we can start loading
# some docs into our Pinecone index:
###################

documents = [
    Document(
        content="""Max Verstappen was the Formula 1 World Drivers' Champion in 2024.
        Lewis Hamilton was the Formula 1 World Drivers' Champion in 2020.
        Nico Rosberg was the Formula 1 World Drivers' Champion in 2016.
        Sebastian Vettel was the Formula 1 World Drivers' Champion in 2013.
        Jenson Button was the Formula 1 World Drivers' Champion in 2009.
        Kimi Räikkönen was the Formula 1 World Drivers' Champion in 2007.
        """,
        metadata=DocMetaData(
            source="wikipedia",
            id="formula-1-facts",
        )
    ),
    Document(
        content="""The Boston Celtics won the NBA Championship for the 2024 NBA season. The MVP for the 2024 NBA Championship was Jaylen Brown.
        The Denver Nuggets won the NBA Championship for the 2023 NBA season. The MVP for the 2023 NBA Championship was Nikola Jokić.
        The Golden State Warriors won the NBA Championship for the 2022 NBA season. The MVP for the 2022 NBA Championship was Stephen Curry.
        The Milwaukee Bucks won the NBA Championship for the 2021 NBA season. The MVP for the 2021 NBA Championship was Giannis Antetokounmpo.
        The Los Angeles Lakers won the NBA Championship for the 2020 NBA season. The MVP for the 2020 NBA Championship was LeBron James.
        The Toronto Raptors won the NBA Championship for the 2019 NBA season. The MVP for the 2019 NBA Championship was Kawhi Leonard.
        """,
        metadata=DocMetaData(
            source="wikipedia",
            id="nba-facts"
        )
    )
]

agent.ingest_docs(documents)

###################
# With the documents now loaded, we can now prompt our agent
###################

formula_one_world_champion_2007 = agent.llm_response(
    message="Who was the Formula 1 World Drivers' Champion in 2007?"
)
try:
    assert "Kimi Räikkönen" in formula_one_world_champion_2007.content
except AssertionError as e:
    print(f"Did not resolve Kimi Räikkönen as the answer, document content: {formula_one_world_champion_2007.content} ")

nba_champion_2023 = agent.llm_response(
    message="Who won the 2023 NBA Championship?"
)
try:
    assert "Denver Nuggets" in nba_champion_2023.content
except AssertionError as e:
    print(f"Did not resolve the Denver Nuggets as the answer, document content: {nba_champion_2023.content}")

nba_mvp_2023 = agent.llm_response(
    message="Who was the MVP for the 2023 NBA Championship?"
)
try:
    assert "Nikola Jokić" in nba_mvp_2023.content
except AssertionError as e:
    print(f"Did not resolve Nikola Jokić as the answer, document content: {nba_mvp_2023.content}")
```
