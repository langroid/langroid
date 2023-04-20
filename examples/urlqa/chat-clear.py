# A version of chat.py that makes all steps clear, with minimal reliance
# on langchain.

# todo:
# [] doc splitting, collecting splits
# [] split -> embedding (via LLM embedding API)
# [] embedding -> vectorstore insertion
# [] query -> vectorstore retrieval of k nearest neighbors
# [] k nearest neighbor vectors -> documents (splits) retrieval
# [] compose query + splits -> LLM query

from llmagent.parsing.urls import get_urls_from_user
from llama_index import GPTFaissIndex
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
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
)
from dotenv import load_dotenv
import os

import hydra
from omegaconf import DictConfig

URLS = [
    "https://www.understandingwar.org/backgrounder/russian-offensive"
    "-campaign-assessment-february-8-2023",
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign"
    "-assessment-february-9-2023",
]


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
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    llm = OpenAI(temperature=0, openai_api_key=api_key)

    text_splitter = CharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # qa = RetrievalQA.from_chain_type(language_models=OpenAI(),
    #                                  chain_type=config.chain_type,
    #                                  retriever=vectorstore.as_retriever())
    #
    # qa.run()

    """
    question = "What is total cost for undergrads at Rutgers"?
    chat_history = [] # later could be [(human,ai)] pairs
    chat_history_str = ""

    Here are the steps
    
    docs = self.retriever.get_relevant_documents(question, k=4) # get relevant splits
    # only take first m docs so we are within limits
    docs = reduce_tokens_below_limit(docs)
    
    chat_history = chat_history_str(chat_history)
    if chat_history:
        # use LLM to rephrase question as stand-alone, given chat_history
        question = LLM.convert_standalone(chat_history, question)
        chat_history = ""
    prompts = [ {"context": d.page_content,
                 "question": question} for d in docs
                ]
    # parallel API calls to get verbatim extracts from each doc, relevant to question
    doc_extracts = LLM.parallel_generate(prompts)
    
    # compose final answer using LLM
    ans = LLM.compose(doc_extracts, question)

    """
    
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
        elif query.startswith(("summar", "?")) and len(response) == 0:
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
