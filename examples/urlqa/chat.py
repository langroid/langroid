# TODO:
# - what happens when query history becomes long?
# - look into summarization of previous responses or context, to ensure we
#    fit into token limit (context-length)
# - monitor our api cost
# - streaming response (i.e. word by word output)
# - make web-ui for this

from llmagent.parsing.urls import get_urls_from_user
from llmagent.prompts.transforms import (
    get_verbatim_extracts,
    get_summary_answer,
    followup_to_standalone
)
from examples.urlqa.config import URLQAConfig
from llmagent.mytypes import Document
from langchain.schema import Document as LDocument
from llmagent.parsing.url_loader import URLLoader
import tiktoken
from halo import Halo
from llmagent.language_models.openai_gpt3 import OpenAIGPT
from llmagent.vector_store.chromadb import ChromaDB
from llmagent.vector_store.qdrantdb import Qdrant
from llmagent.vector_store.faissdb import FAISSDB
from llmagent.utils import configuration
from transformers.utils import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os
from rich import print
import warnings
import hydra
from hydra.core.config_store import ConfigStore

from omegaconf import DictConfig

logging.set_verbosity(logging.ERROR) # for transformers logging


# Register the config with Hydra's ConfigStore
config_dict = URLQAConfig().dict()
cs = ConfigStore.instance()
cs.store(name=URLQAConfig.__name__, node=config_dict)

@hydra.main(version_base=None, config_name=URLQAConfig.__name__)
def main(config: URLQAConfig) -> None:
    configuration.update_global_settings(config, keys=["debug"])

    default_urls = config.urls

    print("[blue]Welcome to the URL chatbot " "[x or q to quit, ? for evidence]")
    print("[blue]Enter some URLs below (or leave empty for default URLs)")
    urls = get_urls_from_user() or default_urls
    loader = URLLoader(urls=urls)
    documents = loader.load()
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    llm = OpenAIGPT(api_key=api_key)
    encoding = tiktoken.encoding_for_model(llm.completion_model)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=config.parsing.chunk_size,
        chunk_overlap=config.parsing.chunk_overlap,
        length_function=lambda text: len(encoding.encode(text)),
    )
    lc_docs = [LDocument(page_content = d.content, metadata = d.metadata)
               for d in documents]
    texts = text_splitter.split_documents(lc_docs)

    # convert texts to list of Documents
    texts = [Document(content=text.page_content, metadata=text.metadata)
             for text in texts]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #embedding_models = OpenAIEmbeddings()

    vecstore_class = dict(faiss = FAISSDB, qdrant = Qdrant, chroma = ChromaDB).get(
        config.vecdb.type, ChromaDB
    )
    vectorstore = vecstore_class.from_documents("urls", texts)


    chat_history = []
    print(f"[green] I have processed the following {len(urls)} URLs:")
    print("\n".join(urls))

    response = dict()

    warnings.filterwarnings(
        "ignore",
        message="Token indices sequence length.*",
        # category=UserWarning,
        module="transformers",
    )

    while True:
        print("\n[blue]Query: ", end="")
        query = input("")
        if query in ["exit", "quit", "q", "x", "bye"]:
            print("[green] Bye, hope this was useful!")
            break
        if query == "?" and len(response) > 0:
            # show evidence for last response
            source = response.metadata["source"]
            if len(source) > 0:
                print("[orange]" + source)
            else:
                print("[orange]No source found")
        elif query.startswith(("summar", "?")) and len(response) == 0:
            # CAUTION: SUMMARIZATION is very expensive: the entire
            # document is sent to the API --
            # this will easily shoot up the number of tokens sent
            print("[green] Summaries not ready, coming soon!")
        else:
            if len(chat_history) > 0:
                with Halo(text="Converting to stand-alone query...",  spinner="dots"):
                    query = followup_to_standalone(llm, chat_history, query)
                print(f"[orange1]New query: {query}")

            with Halo(text="Searching VecDB for relevant doc passages...",
                      spinner="dots"):
                docs_and_scores = vectorstore.similar_texts_with_scores(
                    query, k=4,
                    debug=config.debug
                )
            passages: List[Document] = [
                Document(content=d.content, metadata=d.metadata)
                for (d, s) in docs_and_scores
            ]
            max_score = max([s[1] for s in docs_and_scores])
            with Halo(text="LLM Extracting verbatim passages...",  spinner="dots"):
                verbatim_texts: List[Document] = get_verbatim_extracts(
                    query, passages, llm
                )
            with Halo(text="LLM Generating final answer...",
                      spinner="dots"):
                response = get_summary_answer(query, verbatim_texts, llm, k=4)
            print("[green]relevance = ", max_score)
            print("[green]" + response.content)
            source = response.metadata["source"]
            if len(source) > 0:
                print("[orange]" + source)
            chat_history.append((query, response.content))


if __name__ == "__main__":
    main()
