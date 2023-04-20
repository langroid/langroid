# based on OpenAI-cookbook example

import openai

from typing import List, Iterator
import pandas as pd
import numpy as np
import os
import wget
from ast import literal_eval

import asyncio
from llmagent.parsing.urls import get_urls_from_user
from llmagent.prompts.transforms import(
    get_verbatim_extracts,
    get_summary_answer
)

from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from llmagent.language_models.openai_gpt3 import OpenAIGPT3

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
    llm = OpenAIGPT3(api_key=api_key)

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0,
    )
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    memory = FAISS.from_documents(texts, embeddings)
    query = "What is total cost for undergrad students at CMU?"
    similar_doc_shards = memory.similarity_search(query, k=4)
    passages = [s.page_content for s in similar_doc_shards]
    verbatim_texts = get_verbatim_extracts(query, passages, llm, num_shots=2)

    answer = get_summary_answer(query, verbatim_texts, llm, k=4)
    print(answer)
    # from each shard, get verbatim extract relevant to query, if any
    # use map to apply function to each shard

    #verbatim_extract = map(get_verbatim_extract, similar_doc_shards)


    #print(summary['choices'][0]['text'])

    a = 1


if __name__ == "__main__":
    main()

