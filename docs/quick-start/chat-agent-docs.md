# Augmenting Agents with Retrieval

!!! tip "Script in `langroid-examples`"
A full working example for the material in this section is
in the `chat-agent-docs.py` script in the `langroid-examples` repo:
[`examples/quick-start/chat-agent-docs.py`](https://github.com/langroid/langroid-examples/tree/main/examples/quick-start/chat-agent-docs.py).

## Why is this important?

Until now in this guide, agents have not used external data.
Although LLMs already have enourmous amounts of knowledge "hard-wired"
into their weights during training (and this is after all why ChatGPT
has exploded in popularity), for practical enterprise applications
there are a few reasons it is critical to augment LLMs with access to
specific, external documents:

- **Private data**: LLMs are trained on public data, but in many applications
  we want to use private data that is not available to the public.
  For example, a company may want to extract useful information from its private
  knowledge-base.
- **New data**: LLMs are trained on data that was available at the time of training,
  and so they may not be able to answer questions about new topics
- **Constrained responses, or Grounding**: LLMs are trained to generate text that is
  consistent with the distribution of text in the training data.
  However, in many applications we want to constrain the LLM's responses
  to be consistent with the content of a specific document.
  For example, if we want to use an LLM to generate a response to a customer
  support ticket, we want the response to be consistent with the content of the ticket.
  In other words, we want to reduce the chances that the LLM _hallucinates_
  a response that is not consistent with the ticket.

In all these scenarios, we want to augment the LLM with access to a specific
set of documents, and use _retrieval augmented generation_ (RAG) to generate
more relevant, useful, accurate responses. Langroid provides a simple, flexible mechanism 
RAG using vector-stores, thus ensuring **grounded responses** constrained to 
specific documents. Another key feature of Langroid is that retrieval lineage 
is maintained, and responses based on documents are always accompanied by
**source citations**.

## Langroid's Special Agent: [DocChatAgent]

Langroid provides a special type of agent called 
[`DocChatAgent`](/reference/agent/special/doc_chat_agent), which is a `ChatAgent`
augmented with a vector-store, and some special methods that enable the agent
to ingest documents into the vector-store, and answer queries based on these documents.

The `DocChatAgent` provides many ways to ingest documents into the vector-store,
including from URLs and local file-paths and URLs. Given a collection of document paths,
ingesting their content into the vector-store involves the following steps:

1. Split the document into shards (in a configurable way)
2. Map each shard to an embedding vector using an embedding model. The default
  embedding model is OpenAI's `text-embedding-ada-002` model, but users can 
  instead use `all-MiniLM-L6-v2` from HuggingFace `sentence-transformers` library.[^1]
3. Store embedding vectors in the vector-store, along with the shard's content and 
  any document-level meta-data (this ensures Langroid knows which document a shard
  came from when it retrieves it augment an LLM query)

[^1]: To use the HuggingFace model, you need to install Langroid with the "extras"
option enabled.

`DocChatAgent`'s `llm_response` overrides the default `ChatAgent` method, 
but augmenting the input message with relevant shards from the vector-store,
along with instructions to the LLM to respond based on the shards.

## Example: Answer questions based on documents










