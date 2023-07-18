<div style="display: flex; align-items: center;">
  <img src="docs/assets/orange-logo.png" alt="Logo" 
        width="80" height="80"align="left">
  <h1>Langroid: Harness LLMs with Multi-Agent Programming</h1>
</div>

[![Pytest](https://github.com/langroid/langroid/actions/workflows/pytest.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/pytest.yml)
[![Lint](https://github.com/langroid/langroid/actions/workflows/validate.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/validate.yml)
[![Docs](https://github.com/langroid/langroid/actions/workflows/mkdocs-deploy.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/mkdocs-deploy.yml)

Langroid is an intuitive, lightweight, transparent, flexible, extensible and principled
Python framework to harness LLMs using Multi-Agent Programming (MAP).
We welcome contributions!

Documentation: https://langroid.github.io/langroid/

## Contributors:
- Prasad Chalasani (IIT BTech/CS, CMU PhD/ML; Independent ML Consultant)
- Somesh Jha (IIT BTech/CS, CMU PhD/CS; Professor of CS, U Wisc at Madison)
- Mohannad Alhanahnah (Research Associate, U Wisc at Madison)
- Ashish Hooda (IIT BTech/CS; PhD Candidate, U Wisc at Madison)

## Overview

### The LLM Opportunity

Given the remarkable abilities of recent Large Language Models (LLMs), there
is an unprecedented opportunity to build intelligent applications powered by
this transformative technology. The top question for any enterprise is: how
best to harness the power of LLMs for complex applications? For technical and
practical reasons, building LLM-powered applications is not as simple as
throwing a task at an LLM-system and expecting it to do it.

### Langroid's Multi-Agent Programming Framework

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
Besides Agents, Langroid also provides simple ways to directly interact with  
LLMs and vector-stores.

### Highlights
Highlights of Langroid's features as of July 2023:

- **Agents as first-class citizens:** An Agent is an abstraction that encapsulates LLM conversation state,
  and optionally a vector-store and tools. Agents are the core abstraction in Langroid.
  Agents act as _message transformers_, and by default provide 3 responder methods,  
  one corresponding to each entity: LLM, Agent, User.
- **Tasks:** A Task class wraps an Agent, and gives the agent instructions (or roles, or goals), 
  manages iteration over an Agent's responder methods, 
  and orchestrates multi-agent interactions via hierarchical, recursive
  task-delegation. The `Task.run()` method has the same 
  type-signature as an Agent's responder's methods, and this is key to how 
  a task of an agent can delegate to other sub-tasks.
- **LLM Support**: Langroid supports OpenAI LLMs including GPT-3.5-Turbo,
  GPT-4-0613
- **Caching of LLM prompts, responses:** Langroid uses [Redis](https://redis.com/try-free/) for caching.
- **Vector Store Support**: [Qdrant](https://qdrant.tech/) and [Chroma](https://www.trychroma.com/) are currently supported.
  Vector stores allow for Retrieval-Augmented-Generaation (RAG).
- **Grounding and source-citation:** Access to external documents via vector-stores 
   allows for grounding and source-citation.
- **Observability: Logging and provenance/lineage:** Langroid generates detailed logs of multi-agent interactions and
  and maintains provenance/lineage of messages, so that you can trace back
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

# Usage/quick-start
These are quick teasers to give a glimpse of what you can do with Langroid
and how your code would look. See the 
[`Getting Started Guide`](https://langroid.github.io/langroid/getting_started/)
for more details.

## Install `langroid` 
Use `pip` to install `langroid` (from PyPi) to your virtual environment:
```bash
pip install langroid
```

## Set up environment variables (API keys, etc)

Copy the `.env-template` file to a new file `.env` and 
insert these secrets:
- **OpenAI API** key (required): If you don't have one, see [this OpenAI Page](https://help.openai.com/en/collections/3675940-getting-started-with-openai-api).
- **Qdrant** Vector Store API Key (required for apps that need retrieval from
  documents): Sign up for a free 1GB account at [Qdrant cloud](https://cloud.qdrant.io)
  Alternatively [Chroma](https://docs.trychroma.com/) is also currently supported. 
  We use the local-storage version of Chroma, so there is no need for an API key.
- **GitHub** Personal Access Token (required for apps that need to analyze git
  repos; token-based API calls are less rate-limited). See this
  [GitHub page](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
- **Redis** Password (optional, only needed to cache LLM API responses):
  Redis [offers](https://redis.com/try-free/) a free 30MB Redis account
  which is more than sufficient to try out Langroid and even beyond.
  
```bash
cp .env-template .env
# now edit the .env file, insert your secrets as above
``` 
Your `.env` file should look like this:
```bash
OPENAI_API_KEY=<your key>
GITHUB_ACCESS_TOKEN=<your token>
REDIS_PASSWORD=<your password>
QDRANT_API_KEY=<your key>
```

Currently only OpenAI models are supported. Others will be added later
(Pull Requests welcome!).

## Direct interaction with OpenAI LLM

```python
from langroid.language_models.openai_gpt import ( 
        OpenAIGPTConfig, OpenAIChatModel, OpenAIGPT,
)
from langroid.language_models.base import LLMMessage, Role

cfg = OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4)

mdl = OpenAIGPT(cfg)

messages = [
  LLMMessage(content="You are a helpful assistant",  role=Role.SYSTEM), 
  LLMMessage(content="What is the capital of Ontario?",  role=Role.USER),
],
response = mdl.chat(messages, max_tokens=200)
```

## Define an agent, set up a task, and run it

```python
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig

config = ChatAgentConfig(
    llm = OpenAIGPTConfig(
        chat_model=OpenAIChatModel.GPT4,
    ),
    vecdb=None, # no vector store
)
agent = ChatAgent(config)
# get response from agent's LLM ...
answer = agent.llm_response("What is the capital of Ontario?")
# ... or set up a task..
task = Task(agent, name="Bot") 
task.run() # ... a loop seeking response from Agent, LLM or User at each turn
```

## Three communicating agents

A toy numbers game, where when given a number `n`:
- `repeater_agent`'s LLM simply returns `n`,
- `even_agent`'s LLM returns `n/2` if `n` is even, else says "DO-NOT-KNOW"
- `odd_agent`'s LLM returns `3*n+1` if `n` is odd, else says "DO-NOT-KNOW"

First define the 3 agents, and set up their tasks with instructions:

```python
    config = ChatAgentConfig(
        llm = OpenAIGPTConfig(
            chat_model=OpenAIChatModel.GPT4,
        ),
        vecdb = None,
    )
    repeater_agent = ChatAgent(config)
    repeater_task = Task(
        repeater_agent,
        name = "Repeater",
        system_message="""
        Your job is to repeat whatever number you receive.
        """,
        llm_delegate=True, # LLM takes charge of task
        single_round=False, 
    )
    even_agent = ChatAgent(config)
    even_task = Task(
        even_agent,
        name = "EvenHandler",
        system_message=f"""
        You will be given a number. 
        If it is even, divide by 2 and say the result, nothing else.
        If it is odd, say {NO_ANSWER}
        """,
        single_round=True,  # task done after 1 step() with valid response
    )

    odd_agent = ChatAgent(config)
    odd_task = Task(
        odd_agent,
        name = "OddHandler",
        system_message=f"""
        You will be given a number n. 
        If it is odd, return (n*3+1), say nothing else. 
        If it is even, say {NO_ANSWER}
        """,
        single_round=True,  # task done after 1 step() with valid response
    )
```
Then add the `even_task` and `odd_task` as sub-tasks of `repeater_task`, 
and run the `repeater_task`, kicking it off with a number as input:
```python
    repeater_task.add_sub_task([even_task, odd_task])
    repeater_task.run("3")
```



