<div align="center">
  <img src="docs/assets/langroid-card-lambda-ossem-rust-1200-630.png" alt="Logo" 
        width="400" align="center">
</div>

<div align="center">

[![Pytest](https://github.com/langroid/langroid/actions/workflows/pytest.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/langroid/langroid/branch/main/graph/badge.svg?token=H94BX5F0TE)](https://codecov.io/gh/langroid/langroid)
[![Lint](https://github.com/langroid/langroid/actions/workflows/validate.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/validate.yml)
[![Docs](https://github.com/langroid/langroid/actions/workflows/mkdocs-deploy.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/mkdocs-deploy.yml)
[![Static Badge](https://img.shields.io/badge/Documentation-blue?link=https%3A%2F%2Flangroid.github.io%2Flangroid%2F&link=https%3A%2F%2Flangroid.github.io%2Flangroid%2F)](https://langroid.github.io/langroid)
[![Static Badge](https://img.shields.io/badge/Discord-Orange?link=https%3A%2F%2Fdiscord.gg%2Fg3nAXCbZ&link=https%3A%2F%2Fdiscord.gg%2Fg3nAXCbZ)](https://discord.gg/g3nAXCbZ)

</div>

<h3 align="center">
  <a target="_blank" 
    href="https://langroid.github.io/langroid/" rel="dofollow">
      <strong>Documentation</strong></a>
  &middot;
  <a target="_blank" href="https://github.com/langroid/langroid-examples" rel="dofollow">
      <strong>Examples Repo</strong></a>
  &middot;
  <a target="_blank" href="https://discord.gg/g3nAXCbZ" rel="dofollow">
      <strong>Discord</strong></a>
  &middot;
  <a target="_blank" href="./CONTRIBUTING.md" rel="dofollow">
      <strong>Contributing</strong></a>

  <br />
</h3>

`Langroid` is an intuitive, lightweight, extensible and principled
Python framework to easily build LLM-powered applications. 
You set up Agents, equip them with optional components (LLM, 
vector-store and methods), assign them tasks, and have them 
collaboratively solve a problem by exchanging messages. 
This Multi-Agent paradigm is inspired by the
[Actor Framework](https://en.wikipedia.org/wiki/Actor_model)
(but you do not need to know anything about this!).


# :rocket: Demo
Suppose you want to extract structured information about the key terms 
of a commercial lease document. You can easily do this with Langroid using a two-agent system,
as we show in the [langroid-examples](https://github.com/langroid/langroid-examples/blob/main/examples/docqa/chat_multi_extract.py) repo.
The demo showcases just a few of the many features of Langroid, such as:
- Multi-agent collaboration: `LeaseExtractor` is in charge of the task, and its LLM (GPT4) generates questions 
to be answered by the `DocAgent`.
- Retrieval augmented question-answering, with **source-citation**: `DocAgent` LLM (GPT4) uses retrieval from a vector-store to 
answer the `LeaseExtractor`'s questions, cites the specific excerpt supporting the answer. 
- Function-calling (also known as tool/plugin): When it has all the information it 
needs, the `LeaseExtractor` LLM presents the information in a structured 
format using a Function-call. 

Here is what it looks like in action:

![Demo](docs/assets/demos/lease-extractor-demo.gif)


# :zap: Highlights

- **Agents as first-class citizens:** The [Agent](https://langroid.github.io/langroid/reference/agent/base/#langroid.agent.base.Agent) class encapsulates LLM conversation state,
  and optionally a vector-store and tools. Agents are a core abstraction in Langroid;
  Agents act as _message transformers_, and by default provide 3 _responder_ methods, one corresponding to each entity: LLM, Agent, User.
- **Tasks:** A [Task](https://langroid.github.io/langroid/reference/agent/task/) class wraps an Agent, and gives the agent instructions (or roles, or goals), 
  manages iteration over an Agent's responder methods, 
  and orchestrates multi-agent interactions via hierarchical, recursive
  task-delegation. The `Task.run()` method has the same 
  type-signature as an Agent's responder's methods, and this is key to how 
  a task of an agent can delegate to other sub-tasks: from the point of view of a Task,
  sub-tasks are simply additional responders, to be used in a round-robin fashion 
  after the agent's own responders.
- **Modularity, Reusabilily, Loose coupling:** The `Agent` and `Task` abstractions allow users to design
  Agents with specific skills, wrap them in Tasks, and combine tasks in a flexible way.
- **LLM Support**: Langroid supports OpenAI LLMs including GPT-3.5-Turbo,
  GPT-4-0613
- **Caching of LLM prompts, responses:** Langroid uses [Redis](https://redis.com/try-free/) for caching.
- **Vector-stores**: [Qdrant](https://qdrant.tech/) and [Chroma](https://www.trychroma.com/) are currently supported.
  Vector stores allow for Retrieval-Augmented-Generation (RAG).
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

--- 

# :gear: Installation and Setup

### Install `langroid`
Langroid requires Python 3.11+. We recommend using a virtual environment.
Use `pip` to install `langroid` (from PyPi) to your virtual environment:
```bash
pip install langroid
```
The core Langroid package lets you use OpenAI Embeddings models via their API. 
If you instead want to use the `all-MiniLM-L6-v2` embeddings model
from from HuggingFace, install Langroid like this:
```bash
pip install langroid[hf-embeddings]
```
Note that this will install `torch` and `sentence-transformers` libraries.

### Set up environment variables (API keys, etc)

To get started, all you need is an OpenAI API Key.
If you don't have one, see [this OpenAI Page](https://help.openai.com/en/collections/3675940-getting-started-with-openai-api).
Currently only OpenAI models are supported. Others will be added later
(Pull Requests welcome!).

In the root of the repo, copy the `.env-template` file to a new file `.env`: 
```bash
cp .env-template .env
```
Then insert your OpenAI API Key. 
Your `.env` file should look like this:
```bash
OPENAI_API_KEY=your-key-here-without-quotes
````


<details>
<summary>Optional Setup Instructions (click to expand) </summary>
All of the below are optional and not strictly needed to run any of the examples.

- **Qdrant** Vector Store API Key, URL. This is only required if you want to use Qdrant cloud.
  You can sign up for a free 1GB account at [Qdrant cloud](https://cloud.qdrant.io).
  If you skip setting up these, Langroid will use Qdrant in local-storage mode.
  Alternatively [Chroma](https://docs.trychroma.com/) is also currently supported. 
  We use the local-storage version of Chroma, so there is no need for an API key.
- **Redis** Password, host, port: This is optional, and only needed to cache LLM API responses
  using Redis Cloud. Redis [offers](https://redis.com/try-free/) a free 30MB Redis account
  which is more than sufficient to try out Langroid and even beyond.
  If you don't set up these, Langroid will use a pure-python 
  Redis in-memory cache via the [Fakeredis](https://fakeredis.readthedocs.io/en/latest/) library.
- **Momento** Serverless Caching of LLM API responses (as an alternative to Redis). 
   To use Momento instead of Redis, simply enter your Momento Token in the `.env` file,
   as the value of `MOMENTO_AUTH_TOKEN` (see example file below).
- **GitHub** Personal Access Token (required for apps that need to analyze git
  repos; token-based API calls are less rate-limited). See this
  [GitHub page](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

If you add all of these optional variables, your `.env` file should look like this:
```bash
OPENAI_API_KEY=your-key-here-without-quotes
GITHUB_ACCESS_TOKEN=your-personal-access-token-no-quotes
REDIS_PASSWORD=your-redis-password-no-quotes
REDIS_HOST=your-redis-hostname-no-quotes
REDIS_PORT=your-redis-port-no-quotes
MOMENTO_AUTH_TOKEN=your-momento-token-no-quotes # instead of REDIS* variables
QDRANT_API_KEY=your-key
QDRANT_API_URL=https://your.url.here:6333 # note port number must be included
```
</details>

---

# :tada: Usage Examples

These are quick teasers to give a glimpse of what you can do with Langroid
and how your code would look. 

:warning: The code snippets below are intended to give a flavor of the code
and they are **not** complete runnable examples! For that we encourage you to 
consult the [`langroid-examples`](https://github.com/langroid/langroid-examples) 
repository.

:information_source: The various LLM prompts and instructions in Langroid
have been tested to work well with GPT4.
Switching to GPT3.5-Turbo is easy via a config flag
(e.g., `cfg = OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT3_5_TURBO)`),
and may suffice for some applications, but in general you may see inferior results.


:book: Also see the
[`Getting Started Guide`](https://langroid.github.io/langroid/quick-start/)
for a detailed tutorial.

Click to expand any of the code examples below:

<details>
<summary> <b> Direct interaction with OpenAI LLM </b> </summary>

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
]
response = mdl.chat(messages, max_tokens=200)
print(response.message)
```
</details>

<details>
<summary> <b> Define an agent, set up a task, and run it </b> </summary>

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
# get response from agent's LLM, and put this in an interactive loop...
# answer = agent.llm_response("What is the capital of Ontario?")
  # ... OR instead, set up a task (which has a built-in loop) and run it
task = Task(agent, name="Bot") 
task.run() # ... a loop seeking response from LLM or User at each turn
```
</details>

<details>
<summary><b> Three communicating agents </b></summary>

```python

A toy numbers game, where when given a number `n`:
- `repeater_agent`'s LLM simply returns `n`,
- `even_agent`'s LLM returns `n/2` if `n` is even, else says "DO-NOT-KNOW"
- `odd_agent`'s LLM returns `3*n+1` if `n` is odd, else says "DO-NOT-KNOW"

First define the 3 agents, and set up their tasks with instructions:

```python
from langroid.utils.constants import NO_ANSWER
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
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

</details>

<details>
<summary><b> Simple Tool/Function-calling example </b></summary>

Langroid leverages Pydantic to support OpenAI's
[Function-calling API](https://platform.openai.com/docs/guides/gpt/function-calling)
as well as its own native tools. The benefits are that you don't have to write
any JSON to specify the schema, and also if the LLM hallucinates a malformed
tool syntax, Langroid sends the Pydantic validation error (suitiably sanitized) 
to the LLM so it can fix it!

Simple example: Say the agent has a secret list of numbers, 
and we want the LLM to find the smallest number in the list. 
We want to give the LLM a `probe` tool/function which takes a
single number `n` as argument. The tool handler method in the agent
returns how many numbers in its list are at most `n`.

First define the tool using Langroid's `ToolMessage` class:


```python
from langroid.agent.tool_message import ToolMessage
class ProbeTool(ToolMessage):
  request: str = "probe" # specifies which agent method handles this tool
  purpose: str = """
        To find how many numbers in my list are less than or equal to  
        the <number> you specify.
        """ # description used to instruct the LLM on when/how to use the tool
  number: int  # required argument to the tool
```

Then define a `SpyGameAgent` as a subclass of `ChatAgent`, 
with a method `probe` that handles this tool:

```python
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
class SpyGameAgent(ChatAgent):
  def __init__(self, config: ChatAgentConfig):
    super().__init__(config)
    self.numbers = [3, 4, 8, 11, 15, 25, 40, 80, 90]

  def probe(self, msg: ProbeTool) -> str:
    # return how many numbers in self.numbers are less or equal to msg.number
    return str(len([n for n in self.numbers if n <= msg.number]))
```

We then instantiate the agent and enable it to use and respond to the tool:

```python
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
spy_game_agent = SpyGameAgent(
    ChatAgentConfig(
        name="Spy",
        llm = OpenAIGPTConfig(
            chat_model=OpenAIChatModel.GPT4,
        ),
        vecdb=None,
        use_tools=False, #  don't use Langroid native tool
        use_functions_api=True, # use OpenAI function-call API
    )
)
spy_game_agent.enable_message(ProbeTool)
```

For a full working example see the
[chat-agent-tool.py](https://github.com/langroid/langroid-examples/blob/main/examples/quick-start/chat-agent-tool.py)
script in the `langroid-examples` repo.
</details>

<details>
<summary> <b>Tool/Function-calling to extract structured information from text </b> </summary>

Suppose you want an agent to extract 
the key terms of a lease, from a lease document, as a nested JSON structure.
First define the desired structure via Pydantic models:

```python
from pydantic import BaseModel
class LeasePeriod(BaseModel):
    start_date: str
    end_date: str


class LeaseFinancials(BaseModel):
    monthly_rent: str
    deposit: str

class Lease(BaseModel):
    period: LeasePeriod
    financials: LeaseFinancials
    address: str
```

Then define the `LeaseMessage` tool as a subclass of Langroid's `ToolMessage`.
Note the tool has a required argument `terms` of type `Lease`:

```python
class LeaseMessage(ToolMessage):
    request: str = "lease_info"
    purpose: str = """
        Collect information about a Commercial Lease.
        """
    terms: Lease
```

Then define a `LeaseExtractorAgent` with a method `lease_info` that handles this tool,
instantiate the agent, and enable it to use and respond to this tool:

```python
class LeaseExtractorAgent(ChatAgent):
    def lease_info(self, message: LeaseMessage) -> str:
        print(
            f"""
        DONE! Successfully extracted Lease Info:
        {message.terms}
        """
        )
        return json.dumps(message.terms.dict())
    
lease_extractor_agent = LeaseExtractorAgent(
  ChatAgentConfig(
    llm=OpenAIGPTConfig(),
    use_functions_api=False,
    use_tools=True,
  )
)
lease_extractor_agent.enable_message(LeaseMessage)
```

See the [`chat_multi_extract.py`](https://github.com/langroid/langroid-examples/blob/main/examples/docqa/chat_multi_extract.py)
script in the `langroid-examples` repo for a full working example.
</details>

<details>
<summary><b> Chat with documents (file paths, URLs, etc) </b></summary>

Langroid provides a specialized agent class `DocChatAgent` for this purpose.
It incorporates document sharding, embedding, storage in a vector-DB, 
and retrieval-augmented query-answer generation.
Using this class to chat with a collection of documents is easy.
First create a `DocChatAgentConfig` instance, with a 
`doc_paths` field that specifies the documents to chat with.

```python
from langroid.agent.doc_chat_agent import DocChatAgentConfig
config = DocChatAgentConfig(
  doc_paths = [
    "https://en.wikipedia.org/wiki/Language_model",
    "https://en.wikipedia.org/wiki/N-gram_language_model",
    "/path/to/my/notes-on-language-models.txt",
  ]
  llm = OpenAIGPTConfig(
    chat_model=OpenAIChatModel.GPT4,
  ),
  vecdb=VectorStoreConfig(
    type="qdrant",
  ),
)
```

Then instantiate the `DocChatAgent`, ingest the docs into the vector-store:

```python
agent = DocChatAgent(config)
agent.ingest()
```
Then we can either ask the agent one-off questions,
```python
agent.chat("What is a language model?")
```
or wrap it in a `Task` and run an interactive loop with the user:
```python
from langroid.task import Task
task = Task(agent)
task.run()
```

See full working scripts in the 
[`docqa`](https://github.com/langroid/langroid-examples/tree/main/examples/docqa)
folder of the `langroid-examples` repo.
</details>

---

# :heart: Thank you to our supporters!

[![Stargazers repo roster for @langroid/langroid](https://reporoster.com/stars/langroid/langroid)](https://github.com/langroid/langroid/stargazers)

# Contributors

- Prasad Chalasani (IIT BTech/CS, CMU PhD/ML; Independent ML Consultant)
- Somesh Jha (IIT BTech/CS, CMU PhD/CS; Professor of CS, U Wisc at Madison)
- Mohannad Alhanahnah (Research Associate, U Wisc at Madison)
- Ashish Hooda (IIT BTech/CS; PhD Candidate, U Wisc at Madison)

