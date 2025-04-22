---
title: 'Langroid: Harness LLMs with Multi-Agent Programming'
draft: false
date: 2023-09-03
authors: 
  - pchalasani
categories:
  - langroid
  - llm
comments: true
---

# Langroid: Harness LLMs with Multi-Agent Programming

## The LLM Opportunity

Given the remarkable abilities of recent Large Language Models (LLMs), there
is an unprecedented opportunity to build intelligent applications powered by
this transformative technology. The top question for any enterprise is: how
best to harness the power of LLMs for complex applications? For technical and
practical reasons, building LLM-powered applications is not as simple as
throwing a task at an LLM-system and expecting it to do it.

<!-- more -->


## Langroid's Multi-Agent Programming Framework

Effectively leveraging LLMs at scale requires a *principled programming
framework*. In particular, there is often a need to maintain multiple LLM
conversations, each instructed in different ways, and "responsible" for
different aspects of a task.


An *agent* is a convenient abstraction that encapsulates LLM conversation
state, along with access to long-term memory (vector-stores) and tools (a.k.a functions
or plugins). Thus a **Multi-Agent Programming** framework is a natural fit
for complex LLM-based applications.

> Langroid is the first Python LLM-application framework that was explicitly
designed  with Agents as first-class citizens, and Multi-Agent Programming
as the core  design principle. The framework is inspired by ideas from the
[Actor Framework](https://en.wikipedia.org/wiki/Actor_model).

Langroid allows an intuitive definition of agents, tasks and task-delegation
among agents. There is a principled mechanism to orchestrate multi-agent
collaboration. Agents act as message-transformers, and take turns responding to (and
transforming) the current message. The architecture is lightweight, transparent,
flexible, and allows other types of orchestration to be implemented.
Besides Agents, Langroid also provides simple ways to directly interact with LLMs and vector-stores.


## Highlights
- **Agents as first-class citizens:** The `Agent` class encapsulates LLM conversation state,
  and optionally a vector-store and tools. Agents are a core abstraction in Langroid;
  Agents act as _message transformers_, and by default provide 3 _responder_ methods, one corresponding to each
  entity: LLM, Agent, User.
- **Tasks:** A Task class wraps an Agent, gives the agent instructions (or roles, or goals),
  manages iteration over an Agent's responder methods,
  and orchestrates multi-agent interactions via hierarchical, recursive
  task-delegation. The `Task.run()` method has the same
  type-signature as an Agent's responder's methods, and this is key to how
  a task of an agent can delegate to other sub-tasks: from the point of view of a Task,
  sub-tasks are simply additional responders, to be used in a round-robin fashion
  after the agent's own responders.
- **Modularity, Reusability, Loose coupling:** The `Agent` and `Task` abstractions allow users to design
  Agents with specific skills, wrap them in Tasks, and combine tasks in a flexible way.
- **LLM Support**: Langroid supports OpenAI LLMs including GPT-3.5-Turbo,
  GPT-4.
- **Caching of LLM prompts, responses:** Langroid by default uses [Redis](https://redis.com/try-free/) for caching.
- **Vector-stores**: [Qdrant](https://qdrant.tech/), [Chroma](https://www.trychroma.com/), LanceDB, Pinecone, PostgresDB (PGVector), Weaviate are currently supported.
  Vector stores allow for Retrieval-Augmented-Generaation (RAG).
- **Grounding and source-citation:** Access to external documents via vector-stores
  allows for grounding and source-citation.
- **Observability, Logging, Lineage:** Langroid generates detailed logs of multi-agent interactions and
  maintains provenance/lineage of messages, so that you can trace back
  the origin of a message.
- **Tools/Plugins/Function-calling**: Langroid supports OpenAI's recently
  released [function calling](https://platform.openai.com/docs/guides/gpt/function-calling)
  feature. In addition, Langroid has its own native equivalent, which we
  call **tools** (also known as "plugins" in other contexts). Function
  calling and tools have the same developer-facing interface, implemented
  using [Pydantic](https://docs.pydantic.dev/latest/),
  which makes it very easy to define tools/functions and enable agents
  to use them. Benefits of using Pydantic are that you never have to write
  complex JSON specs for function calling, and when the LLM
  hallucinates malformed JSON, the Pydantic error message is sent back to
  the LLM so it can fix it!

<iframe src="https://langroid.substack.com/embed" width="480" height="320" style="border:1px solid #EEE; background:white;" frameborder="0" scrolling="no"></iframe>
