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
more relevant, useful responses. Langroid provides a simple, flexible mechanism 
RAG using vector-stores.










