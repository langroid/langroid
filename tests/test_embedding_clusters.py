import pytest
from llmagent.mytypes import Document
from llmagent.embedding_models.clustering import densest_doc_clusters
from dotenv import load_dotenv
from llmagent.embedding_models.base import EmbeddingModel
from llmagent.embedding_models.models import (
    OpenAIEmbeddingsConfig,
    SentenceTransformerEmbeddingsConfig,
)
import os


load_dotenv()
openai_cfg = OpenAIEmbeddingsConfig(
    model_type="openai",
    model_name="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY"),
    dims=1536,
)

sentence_cfg = SentenceTransformerEmbeddingsConfig(
    model_type="sentence-transformer",
    model_name="all-MiniLM-L6-v2",
    dims=384,
)

openai_model = EmbeddingModel.create(openai_cfg)
sentence_model = EmbeddingModel.create(sentence_cfg)

openai_fn = openai_model.embedding_fn()
sentence_fn = sentence_model.embedding_fn()


@pytest.mark.parametrize("emb_fn", [openai_fn, sentence_fn])
def test_densest_doc_clusters(emb_fn):
    # Create realistic sample documents
    technology_docs = [
        Document(
            content="""Artificial intelligence (AI) is a fast-growing field that
                    aims to create intelligent machines capable of performing tasks
                    without human intervention. These tasks include learning, problem-solving,
                    perception, and natural language understanding. Machine learning, a subset of AI,
                    involves the development of algorithms that allow computers to learn from and make
                    decisions based on data.""",
            metadata={"id": 1, "topic": "AI"},
        ),
        Document(
            content="""Virtual reality (VR) is an immersive technology that can transport users
                    into simulated environments. By wearing a VR headset, users can experience
                    a sense of presence and interact with the virtual world using handheld controllers
                    or motion tracking. VR has applications in gaming, training, education, and
                    healthcare, among other fields.""",
            metadata={"id": 2, "topic": "VR"},
        ),
        Document(
            content="""The Internet of Things (IoT) is a network of interconnected physical devices
                    that collect and exchange data. IoT devices are embedded with sensors, software,
                    and connectivity, allowing them to collect and transmit data to a central
                    system for analysis. This technology has applications in various industries,
                    such as smart homes, transportation, agriculture, and manufacturing.""",
            metadata={"id": 3, "topic": "IOT"},
        ),
    ]

    sports_docs = [
        Document(
            content="""Soccer, also known as football, is a popular sport played by two teams
                    of eleven players, with each team trying to score by getting a ball into the
                    opponent's goal. The game is played on a rectangular field with a goal at each
                    end. The players use any part of their body except their hands and arms to
                    manipulate the ball.""",
            metadata={"id": 4, "topic": "soccer"},
        ),
        Document(
            content="""Basketball is a fast-paced team sport played on a rectangular court with
                    two teams of five players each. The objective is to score points by shooting a
                    ball through a hoop mounted at a height of 10 feet on a backboard at each end
                    of the court. The team with the most points at the end of the game wins.""",
            metadata={"id": 5, "topic": "basketball"},
        ),
        Document(
            content="""Tennis is a racket sport that can be played individually (singles) or
                    between two teams of two players each (doubles). The goal is to hit a ball
                    over a net and into the opponent's court in such a way that the opponent cannot
                    return it. Players score points by winning individual rallies, and a match is won
                    by winning a predetermined number of sets. Tennis is played on various surfaces,
                    such as grass, clay, and hard courts.""",
            metadata={"id": 6, "topic": "tennis"},
        ),
    ]

    docs = technology_docs * 10 + sports_docs * 10
    # assign unique ids
    for i, doc in enumerate(docs):
        doc.metadata["id"] = i
    # Number of densest clusters to find
    k = 6

    # Call densest_doc_clusters function
    representative_docs = densest_doc_clusters(docs, k, emb_fn)

    # Check if the output is a list
    assert isinstance(representative_docs, list)

    # Check if the output list has the correct length
    assert len(representative_docs) <= k
    assert len(representative_docs) >= 2

    # Check if the output list contains Document instances
    assert all(isinstance(doc, Document) for doc in representative_docs)

    # Check if the representative documents have different topics
    # assert (
    #     representative_docs[0].metadata["topic"]
    #     != representative_docs[1].metadata["topic"]
    # )
