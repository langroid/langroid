from rich.console import Console
from rich import print
from examples.urlqa.doc_chat_agent import DocChatAgentConfig, DocChatAgent
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.parsing.parser import ParsingConfig, Splitter
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.mytypes import Document, DocMetaData
from llmagent.agent.chat_document import ChatDocument, ChatDocMetaData
from llmagent.agent.base import Entity
from llmagent.utils.constants import NO_ANSWER
from typing import List, Optional
from abc import ABC, abstractmethod


console = Console()


class RecordMetadata(DocMetaData):
    id: int | str = None


class RecordDoc(Document):
    metadata: RecordMetadata


class RetrieverAgentConfig(DocChatAgentConfig):
    n_matches: int = 3
    debug: bool = False
    max_context_tokens = 500
    conversation_mode = True
    cache: bool = True  # cache results
    gpt4: bool = True  # use GPT-4
    stream: bool = True  # allow streaming where needed
    max_tokens: int = 10000
    vecdb: VectorStoreConfig = QdrantDBConfig(
        type="qdrant",
        collection_name=None,
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


class RetrieverAgent(DocChatAgent, ABC):
    """
    Agent for retrieving whole records/docs matching a query
    """

    def __init__(self, config: RetrieverAgentConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def get_records(self) -> List[RecordDoc]:
        pass

    def ingest(self) -> None:
        records = self.get_records()
        self.vecdb.add_documents(records)

    def llm_response(
        self,
        query: str | ChatDocument = None,
    ) -> Optional[ChatDocument]:
        if not self.llm_can_respond(query):
            return None
        if query is None:
            return super().llm_response(None)
        if isinstance(query, ChatDocument):
            query_str = query.content
        else:
            query_str = query
        docs = self.get_relevant_docs(query_str)
        if len(docs) == 0:
            return None
        content = "\n\n".join([d.content for d in docs])
        print(f"[green]{content}")
        meta = dict(
            sender=Entity.LLM,
        )
        meta.update(docs[0].metadata)

        return ChatDocument(
            content=content,
            metadata=ChatDocMetaData(**meta),
        )

    def get_nearest_docs(self, query: str) -> List[Document]:
        """
        Given a query, get the records/docs whose contents are closest to the
            query, in terms of vector similarity.
        Args:
            query: query string
        Returns:
            list of Document objects
        """
        with console.status("[cyan]Searching VecDB for similar docs/records..."):
            docs_and_scores = self.vecdb.similar_texts_with_scores(
                query,
                k=self.config.parsing.n_similar_docs,
            )
        docs: List[Document] = [
            Document(content=d.content, metadata=d.metadata)
            for (d, _) in docs_and_scores
        ]
        return docs

    def get_relevant_docs(self, query: str) -> List[Document]:
        """
        Given a query, get the records/docs whose contents are most relevant to the
            query. First get nearest docs from vector store, then select the best
            matches according to the LLM.
        Args:
            query (str): query string

        Returns:
            List[Document]: list of Document objects
        """
        response = Document(
            content=NO_ANSWER,
            metadata=DocMetaData(
                source="None",
            ),
        )
        nearest_docs = self.get_nearest_docs(query)
        if len(nearest_docs) == 0:
            return [response]
        with console.status("LLM selecting relevant docs from retrieved ones..."):
            response = self.llm_select_relevant_docs(query, nearest_docs)

        return response

    def llm_select_relevant_docs(
        self, query: str, docs: List[Document]
    ) -> List[Document]:
        """
        Given a query and a list of docs, select the docs whose contents match best,
            according to the LLM. Use the doc IDs to select the docs from the vector
            store.
        Args:
            query: query string
            docs: list of Document objects
        Returns:
            list of Document objects
        """
        doc_contents = "\n\n".join(
            [f"DOC: ID={d.id()}, content={d.content}" for d in docs]
        )
        prompt = f"""
        Given the following QUERY: 
        {query}
        and the following DOCS with IDs and contents
        {doc_contents}
        
        Find at most {self.config.n_matches} DOCs that are most relevant to the QUERY.
        Return your as a sequence of DOC IDS ONLY, for example: 
        "id1 id2 id3..."
        """
        default_response = Document(
            content=NO_ANSWER,
            metadata=DocMetaData(
                source="None",
            ),
        )

        response = self.llm.generate(
            prompt, max_tokens=self.config.llm.max_output_tokens
        )
        if response.message == NO_ANSWER:
            return [default_response]
        ids = response.message.split()
        if len(ids) == 0:
            return [default_response]
        docs = self.vecdb.get_documents_by_ids(ids)
        return [
            Document(content=d.content, metadata=DocMetaData(source="LLM"))
            for d in docs
        ]
