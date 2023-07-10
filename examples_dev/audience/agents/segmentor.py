import csv
from typing import List

from examples_dev.audience.agents.retriever import (
    RecordDoc,
    RecordMetadata,
    RetrieverAgent,
    RetrieverAgentConfig,
)
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.parsing.parser import ParsingConfig, Splitter
from langroid.prompts.prompts_config import PromptsConfig
from langroid.vector_store.base import VectorStoreConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig


class AudienceTaxonomyMetadata(RecordMetadata):
    parent: str = None
    condensed_name: str = None
    tiers: List[str] = None


class AudienceTaxonomyRecord(RecordDoc):
    metadata: AudienceTaxonomyMetadata


class SegmentorConfig(RetrieverAgentConfig):
    system_message: str = "You are a marketing expert"
    user_message: str = """
        Your task is to map audience description to a standard taxonomy.
        You will be given a description of an audience segment and you will need to
        map it to one or more categories in the standard IAB taxonomy.
    """
    filename: str = None  # contains taxonomy records
    n_tiers: int = 6
    debug: bool = False
    max_context_tokens = 500
    conversation_mode = True
    cache: bool = True  # cache results
    gpt4: bool = False  # use GPT-4
    stream: bool = True  # allow streaming where needed
    max_tokens: int = 10000
    vecdb: VectorStoreConfig = QdrantDBConfig(
        type="qdrant",
        collection_name="IAB-audience-taxonomy",
        storage_path=".qdrant/data/",
        embedding=OpenAIEmbeddingsConfig(
            model_type="openai",
            model_name="text-embedding-ada-002",
            dims=1536,
        ),
    )

    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
    )
    parsing: ParsingConfig = ParsingConfig(
        splitter=Splitter.TOKENS,
        chunk_size=100,
        n_similar_docs=5,
    )

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )


class Segmentor(RetrieverAgent):
    """
    Agent for extracting audience segment records matching a query
    """

    def __init__(self, config: SegmentorConfig):
        super().__init__(config)
        self.config = config

    def get_records(self) -> List[AudienceTaxonomyRecord]:
        if self.config.filename is None:
            return []
        taxonomy_records = []
        with open(self.config.filename, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            row_num = 0
            for row in reader:
                tiers = [f"Tier {t+1}" for t in range(self.config.n_tiers)]
                full_path = "|".join(row[t] for t in tiers if row[t] != "")
                row_num += 1
                if row["Unique ID"] == "":
                    continue
                meta = AudienceTaxonomyMetadata(
                    source=f"IAB_row_{row_num}",
                    id=int(row["Unique ID"]),
                    parent=row["Parent ID"],
                    condensed_name=list(row.values())[2],
                )
                taxonomy_records.append(
                    AudienceTaxonomyRecord(
                        content=full_path,
                        metadata=meta,
                    )
                )
        return taxonomy_records
