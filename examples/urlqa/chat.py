# query URL(s) using langchain utilities

# TODO:
# see chat_vector_db.ipynb in langchain for reference.
# (1) expose the prompt and allow customizing it, so we can explicitly see
#   how context, chat-history, and query are combined
# (2) what happens when query history becomes long?
# (3) look into summarization of previous responses or context, to ensure we
#    fit into token limit (context-length)
# (4) monitor our api cost
# (5) response should show "source_documents" in addition to "answer"
# (6) streaming response (i.e. word by word output)
# (7) make web-ui for this


from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import (
    ConversationalRetrievalChain,
    #    RetrievalQA,
)

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.conversational_retrieval.prompts import \
    CONDENSE_QUESTION_PROMPT

import hydra
from omegaconf import DictConfig

URLS = [
    "https://www.understandingwar.org/backgrounder/russian-offensive"
    "-campaign-assessment-february-8-2023",
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign"
    "-assessment-february-9-2023",
]


def get_urls_from_user():
    # Create an empty set to store the URLs.
    url_set = set()

    # Use a while loop to continuously ask the user for URLs.
    while True:
        # Prompt the user for input.
        url = input("Enter a URL (type 'done' or hit return to finish): ")

        # Check if the user wants to exit the loop.
        if url.lower() == "done" or url == "":
            break

        # Add the URL to the set.
        url_set.add(url)

    return url_set


def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)


@hydra.main(version_base=None, config_path="../configs", config_name="params")
def main(config: DictConfig) -> None:
    config = config.settings
    print(config)
    debug = config.debug
    default_urls = config.get("urls", URLS)
    urls = get_urls_from_user() or default_urls
    loader = UnstructuredURLLoader(urls=urls)
    # loader = SeleniumURLLoader(urls=urls)
    documents = loader.load()
    llm = OpenAI(temperature=0)

    text_splitter = CharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # qa = RetrievalQA.from_chain_type(llm=OpenAI(),
    #                                  chain_type=config.chain_type,
    #                                  retriever=vectorstore.as_retriever())
    #
    # qa.run()

    question_generator = LLMChain(
        llm=llm,
        prompt=CONDENSE_QUESTION_PROMPT,
        verbose=debug,
    )
    doc_chain = load_qa_with_sources_chain(
        llm,
        chain_type="map_reduce",
        verbose=debug,
    )

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
    )

    print("Welcome to the URL chatbot [x or q to quit, ? for explanation]")
    chat_history = []
    print(f"I have processed the following {len(urls)} URLs:")
    print("\n".join(urls))

    response = dict()

    while True:
        query = input("Query: ")
        if query in ["exit", "quit", "q", "x", "bye"]:
            print("Bye, hope this was useful!")
            break
        if query == "?" and len(response) > 0:
            # show evidence for last response
            source_doc0 = response["source_documents"][0]
            print("Source document content:")
            print(source_doc0.page_content)
            print(f"Source document URL: {source_doc0.metadata['source']}")
        elif (query.startswith("sumar") or query.startswith("?")) and len(
            response
        ) == 0:
            # CAUTION: SUMMARIZATION is very expensive: the entire
            # document is sent to the API --
            # this will easily shoot up the number of tokens sent
            print("The summary of these urls is:")
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                verbose=debug,
            )
            chain.verbose = debug
            summary = chain.run(documents)
            print(summary)
        else:
            response = qa(dict(question=query, chat_history=chat_history))
            chat_history.append((query, response["answer"]))
            print(response["answer"])


if __name__ == "__main__":
    main()
