from typing import List
from examples.urlqa.doc_chat_agent import DocChatAgent, DocChatAgentConfig
from llmagent.parsing.repo_loader import RepoLoader, RepoLoaderConfig
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.parsing.parser import ParsingConfig, Splitter
from llmagent.parsing.code_parser import CodeParsingConfig
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.prompts.templates import ANSWER_PROMPT_USE_HISTORY_GPT4
from llmagent.mytypes import Document, DocMetaData

import os
from rich import print
from rich.console import Console

console = Console()


DEFAULT_CODE_CHAT_INSTRUCTIONS = """
Your task is to answer questions about a code repository. You will be given directly 
listings, text and code from various files in the repository. You must answer based 
on the information given to you. If you are asked to see if there is a certain file 
in the repository, and it does not occur in the listings you are shown, then you can 
simply answer "No".
"""

DEFAULT_CODE_CHAT_SYSTEM_MESSAGE = """
You are an expert software engineer, helping me understand a code repository.
"""

DEFAULT_CODE_CHAT_ANSWER_PROMPT_GPT4 = """
        Use either the extracts from the code repo provided below, OR any of the info 
        shown previously, to answer the question below. If there's not enough 
        information, respond with "I don't know." 
        Compose your complete answer and cite all supporting sources on a separate 
        separate line as "SOURCE:". 
        
        {extracts}
        
        {question}
        Answer:   
""".strip()


class CodeChatAgentConfig(DocChatAgentConfig):
    system_message: str = DEFAULT_CODE_CHAT_SYSTEM_MESSAGE
    user_message: str = DEFAULT_CODE_CHAT_INSTRUCTIONS
    # threshold to decide whether to extract relevant parts
    summarize_prompt: str = ANSWER_PROMPT_USE_HISTORY_GPT4
    max_context_tokens: int = 500
    conversation_mode: bool = True
    content_includes: List[str] = ["txt", "md", "yml", "yaml", "sh", "Makefile"]
    content_excludes: List[str] = []
    repo_url: str = "https://github.com/eugeneyan/testing-ml"
    gpt4: bool = False
    cache: bool = True
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    vecdb: VectorStoreConfig = QdrantDBConfig(
        type="qdrant",
        collection_name="llmagent-codechat",
        storage_path=".qdrant/codechat/",
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
        splitter=Splitter.TOKENS,
        chunk_size=100,
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


class CodeChatAgent(DocChatAgent):
    """
    Agent for chatting with a code repository.
    """

    def __init__(self, config: CodeChatAgentConfig):
        super().__init__(config)
        self.original_docs: List[Document] = None
        repo_loader = RepoLoader(self.config.repo_url, config=RepoLoaderConfig())

        repo_tree, _ = repo_loader.load(depth=1, lines=20)
        repo_listing = "\n".join(repo_loader.ls(repo_tree, depth=1))

        repo_info_message = f"""
        Here is some information about the code repository that you can use, 
        in the subsequent questions. For any future questions, you can refer back to 
        this info if needed.
        
        Here is a listing of the files and directories at the root of the repo:
        {repo_listing}
        """

        self.add_user_message(repo_info_message)

        dct, documents = repo_loader.load(depth=2, lines=100)
        listing = (
            [
                """
                      List of ALL files and directories in this project:
                      If a file is not in this list, then we can be sure that
                      it is not in the repo!
                      """
            ]
            + repo_loader.ls(dct, depth=1)
        )
        listing = Document(
            content="\n".join(listing),
            metadata=DocMetaData(source="repo_listing"),
        )

        code_docs = [
            doc
            for doc in documents
            if doc.metadata.language not in (["md", "txt"] + config.content_excludes)
        ] + [listing]

        text_docs = [doc for doc in documents if doc.metadata.language in ["md", "txt"]]

        with console.status("Processing code repo..."):
            self.config.parsing = config.parsing
            n_text_splits = self.ingest_docs(text_docs)
            self.config.parsing = config.code_parsing
            n_code_splits = self.ingest_docs(code_docs)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        print(
            f"""
        [green]I have processed {len(documents)} files from the following GitHub Repo into 
        {n_text_splits} text chunks and {n_code_splits} code chunks:
        {self.config.repo_url}
        """.strip()
        )
