<div align="center">
  <img src="docs/assets/langroid-card-lambda-ossem-rust-1200-630.png" alt="Logo" 
        width="400" align="center">
</div>

<div align="center">

[![PyPI - Version](https://img.shields.io/pypi/v/langroid)](https://pypi.org/project/langroid/)
[![Pytest](https://github.com/langroid/langroid/actions/workflows/pytest.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/langroid/langroid/branch/main/graph/badge.svg?token=H94BX5F0TE)](https://codecov.io/gh/langroid/langroid)
[![Lint](https://github.com/langroid/langroid/actions/workflows/validate.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/validate.yml)
[![Docs](https://github.com/langroid/langroid/actions/workflows/mkdocs-deploy.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/mkdocs-deploy.yml)

[![Static Badge](https://img.shields.io/badge/Documentation-blue?link=https%3A%2F%2Flangroid.github.io%2Flangroid%2F&link=https%3A%2F%2Flangroid.github.io%2Flangroid%2F)](https://langroid.github.io/langroid)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/ZU36McDgDs)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langroid/langroid/blob/main/examples/langroid_quick_examples.ipynb)

[![Docker Pulls](https://img.shields.io/docker/pulls/langroid/langroid.svg)](https://hub.docker.com/r/langroid/langroid)
![Docker Image Size (tag)](https://img.shields.io/docker/image-size/langroid/langroid/latest)
[![Multi-Architecture DockerHub](https://github.com/langroid/langroid/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/docker-publish.yml)

[![Substack](https://img.shields.io/badge/Substack-%23006f5c.svg?style=for-the-badge&logo=substack&logoColor=FF6719)](https://langroid.substack.com/p/langroid-harness-llms-with-multi-agent-programming)

[![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Flangroid%2Flangroid&t=Harness%20LLMs%20with%20Multi-Agent%20Programming)
[![Share on Reddit](https://img.shields.io/badge/Reddit-FF4500?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Flangroid%2Flangroid&title=Harness%20LLMs%20with%20Multi-Agent%20Programming)
[![Share on Twitter](https://img.shields.io/twitter/url?style=social&url=https://github.com/langroid/langroid)](https://twitter.com/intent/tweet?text=Langroid%20is%20a%20powerful,%20elegant%20new%20framework%20to%20easily%20build%20%23LLM%20applications.%20You%20set%20up%20LLM-powered%20Agents%20with%20vector-stores,%20assign%20tasks,%20and%20have%20them%20collaboratively%20solve%20problems%20via%20message-transformations.%20https://github.com/langroid/langroid)
[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/shareArticle?mini=true&url=https://github.com/langroid/langroid&title=Langroid:%20A%20Powerful,%20Elegant%20Framework&summary=Langroid%20is%20a%20powerful,%20elegant%20new%20framework%20to%20easily%20build%20%23LLM%20applications.%20You%20set%20up%20LLM-powered%20Agents%20with%20vector-stores,%20assign%20tasks,%20and%20have%20them%20collaboratively%20solve%20problems%20via%20message-transformations.)


</div>

<h3 align="center">
  <a target="_blank" 
    href="https://langroid.github.io/langroid/" rel="dofollow">
      <strong>Documentation</strong></a>
  &middot;
  <a target="_blank" href="https://github.com/langroid/langroid-examples" rel="dofollow">
      <strong>Examples Repo</strong></a>
  &middot;
  <a target="_blank" href="https://discord.gg/ZU36McDgDs" rel="dofollow">
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

`Langroid` is a fresh take on LLM app-development, where considerable thought has gone 
into simplifying the developer experience; it does not use `Langchain`.

We welcome contributions -- See the [contributions](./CONTRIBUTING.md) document
for ideas on what to contribute.

**Questions, Feedback, Ideas? Join us on [Discord](https://discord.gg/ZU36McDgDs)!**

<details>
<summary> <b>:fire: Updates/Releases</b></summary>

- **Oct 2023:**
  - **0.1.84:** Added [LiteLLM](https://docs.litellm.ai/docs/providers), so now Langroid can be used with over 100 LLM providers (remote or local)! 
     See guide [here](https://langroid.github.io/langroid/tutorials/non-openai-llms/).
- **Sep 2023:**
  - **0.1.78:** Async versions of several Task, Agent and LLM methods; 
      Nested Pydantic classes are now supported for LLM Function-calling, Tools, Structured Output.    
  - **0.1.76:** DocChatAgent: support for loading `docx` files (preliminary).
  - **0.1.72:** Many improvements to DocChatAgent: better embedding model, 
          hybrid search to improve retrieval, better pdf parsing, re-ranking retrieved results with cross-encoders. 
  - **Use with local LLama Models:** see tutorial [here](https://langroid.github.io/langroid/blog/2023/09/14/using-langroid-with-local-llms/)
  - **Langroid Blog/Newsletter Launched!**: First post is [here](https://substack.com/notes/post/p-136704592) -- Please subscribe to stay updated. 
  - **0.1.56:** Support Azure OpenAI. 
  - **0.1.55:** Improved [`SQLChatAgent`](https://github.com/langroid/langroid/blob/main/langroid/agent/special/sql/sql_chat_agent.py) that efficiently retrieves relevant schema info when translating natural language to SQL.  
- **Aug 2023:**
  - **[Hierarchical computation](https://langroid.github.io/langroid/examples/agent-tree/)** example using Langroid agents and task orchestration.
  - **0.1.51:** Support for global state, see [test_global_state.py](tests/main/test_global_state.py).
  - **:whale: Langroid Docker image**, available, see instructions below.
  - [**RecipientTool**](langroid/agent/tools/recipient_tool.py) enables (+ enforces) LLM to 
specify an intended recipient when talking to 2 or more agents. 
See [this test](tests/main/test_recipient_tool.py) for example usage.
  - **Example:** [Answer questions](examples/docqa/chat-search.py) using Google Search + vecdb-retrieval from URL contents. 
  - **0.1.39:** [`GoogleSearchTool`](langroid/agent/tools/google_search_tool.py) to enable Agents (their LLM) to do Google searches via function-calling/tools.
    See [this chat example](examples/basic/chat-search.py) for how easy it is to add this tool to an agent.
  - **Colab notebook** to try the quick-start examples: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langroid/langroid/blob/main/examples/langroid_quick_examples.ipynb) 
  - **0.1.37:** Added [`SQLChatAgent`](langroid/agent/special/sql_chat_agent.py) -- thanks to our latest contributor [Rithwik Babu](https://github.com/rithwikbabu)!
  - Multi-agent Example: [Autocorrect chat](examples/basic/autocorrect.py)
- **July 2023:** 
  - **0.1.30:** Added [`TableChatAgent`](langroid/agent/special/table_chat_agent.py) to 
    [chat](examples/data-qa/table_chat.py) with tabular datasets (dataframes, files, URLs): LLM generates Pandas code,
    and code is executed using Langroid's tool/function-call mechanism. 
  - **Demo:** 3-agent system for Audience [Targeting](https://langroid.github.io/langroid/demos/targeting/audience-targeting/).
  - **0.1.27**: Added [support](langroid/cachedb/momento_cachedb.py) 
    for [Momento Serverless Cache](https://www.gomomento.com/) as an alternative to Redis.
  - **0.1.24**: [`DocChatAgent`](langroid/agent/special/doc_chat_agent.py) 
    now [accepts](langroid/parsing/document_parser.py) PDF files or URLs.

</details>

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

Here is what it looks like in action 
(a pausable mp4 video is [here](https://vimeo.com/871429249)).

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
  GPT-4.
- **Caching of LLM responses:** Langroid supports [Redis](https://redis.com/try-free/) and 
  [Momento](https://www.gomomento.com/) to cache LLM responses.
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
If you instead want to use the `sentence-transformers` embedding models from HuggingFace, 
install Langroid like this: 
```bash
pip install langroid[hf-embeddings]
```

<details>
<summary><b>Optional Installs for using SQL Chat with a PostgreSQL DB </b></summary>

If you are using `SQLChatAgent` 
(e.g. the script [`examples/data-qa/sql-chat/sql_chat.py`](examples/data-qa/sql-chat/sql_chat.py)),
with a postgres db, you will need to:

- Install PostgreSQL dev libraries for your platform, e.g.
  - `sudo apt-get install libpq-dev` on Ubuntu,
  - `brew install postgresql` on Mac, etc.
- Install langroid with the postgres extra, e.g. `pip install langroid[postgres]`
  or `poetry add langroid[postgres]` or `poetry install -E postgres`.
  If this gives you an error, try `pip install psycopg2-binary` in your virtualenv.
</details>

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

Alternatively, you can set this as an environment variable in your shell
(you will need to do this every time you open a new shell):
```bash
export OPENAI_API_KEY=your-key-here-without-quotes
```


<details>
<summary><b>Optional Setup Instructions (click to expand) </b></summary>

All of the following environment variable settings are optional, and some are only needed 
to use specific features (as noted below).

- **Qdrant** Vector Store API Key, URL. This is only required if you want to use Qdrant cloud.
  You can sign up for a free 1GB account at [Qdrant cloud](https://cloud.qdrant.io).
  If you skip setting up these, Langroid will use Qdrant in local-storage mode.
  Alternatively [Chroma](https://docs.trychroma.com/) is also currently supported. 
  We use the local-storage version of Chroma, so there is no need for an API key.
  Langroid uses Qdrant by default.
- **Redis** Password, host, port: This is optional, and only needed to cache LLM API responses
  using Redis Cloud. Redis [offers](https://redis.com/try-free/) a free 30MB Redis account
  which is more than sufficient to try out Langroid and even beyond.
  If you don't set up these, Langroid will use a pure-python 
  Redis in-memory cache via the [Fakeredis](https://fakeredis.readthedocs.io/en/latest/) library.
- **Momento** Serverless Caching of LLM API responses (as an alternative to Redis). 
   To use Momento instead of Redis:
  - enter your Momento Token in the `.env` file, as the value of `MOMENTO_AUTH_TOKEN` (see example file below),
  - in the `.env` file set `CACHE_TYPE=momento` (instead of `CACHE_TYPE=redis` which is the default).
- **GitHub** Personal Access Token (required for apps that need to analyze git
  repos; token-based API calls are less rate-limited). See this
  [GitHub page](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
- **Google Custom Search API Credentials:** Only needed to enable an Agent to use the `GoogleSearchTool`.
  To use Google Search as an LLM Tool/Plugin/function-call, 
  you'll need to set up 
  [a Google API key](https://developers.google.com/custom-search/v1/introduction#identify_your_application_to_google_with_api_key),
  then [setup a Google Custom Search Engine (CSE) and get the CSE ID](https://developers.google.com/custom-search/docs/tutorial/creatingcse).
  (Documentation for these can be challenging, we suggest asking GPT4 for a step-by-step guide.)
  After obtaining these credentials, store them as values of 
  `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` in your `.env` file. 
  Full documentation on using this (and other such "stateless" tools) is coming soon, but 
  in the meantime take a peek at this [chat example](examples/basic/chat-search.py), which 
  shows how you can easily equip an Agent with a `GoogleSearchtool`.
  


If you add all of these optional variables, your `.env` file should look like this:
```bash
OPENAI_API_KEY=your-key-here-without-quotes
GITHUB_ACCESS_TOKEN=your-personal-access-token-no-quotes
CACHE_TYPE=redis # or momento
REDIS_PASSWORD=your-redis-password-no-quotes
REDIS_HOST=your-redis-hostname-no-quotes
REDIS_PORT=your-redis-port-no-quotes
MOMENTO_AUTH_TOKEN=your-momento-token-no-quotes # instead of REDIS* variables
QDRANT_API_KEY=your-key
QDRANT_API_URL=https://your.url.here:6333 # note port number must be included
GOOGLE_API_KEY=your-key
GOOGLE_CSE_ID=your-cse-id
```
</details>

<details>
<summary><b>Optional setup instructions for Microsoft Azure OpenAI(click to expand)</b></summary> 

When using Azure OpenAI, additional environment variables are required in the 
`.env` file.
This page [Microsoft Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?tabs=command-line&pivots=programming-language-python#environment-variables)
provides more information, and you can set each environment variable as follows:

- `AZURE_API_KEY`, from the value of `API_KEY`
- `AZURE_OPENAI_API_BASE` from the value of `ENDPOINT`, typically looks like `https://your.domain.azure.com`.
- For `AZURE_OPENAI_API_VERSION`, you can use the default value in `.env-template`, and latest version can be found [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new#azure-openai-chat-completion-general-availability-ga)
- `AZURE_OPENAI_DEPLOYMENT_NAME` is the name of the deployed model, which is defined by the user during the model setup 
- `AZURE_GPT_MODEL_NAME` GPT-3.5-Turbo or GPT-4 model names that you chose when you setup your Azure OpenAI account.

</details>

---

# :whale: Docker Instructions

We provide a containerized version of the [`langroid-examples`](https://github.com/langroid/langroid-examples) 
repository via this [Docker Image](https://hub.docker.com/r/langroid/langroid).
All you need to do is set up environment variables in the `.env` file.
Please follow these steps to setup the container:

```bash
# get the .env file template from `langroid` repo
wget https://github.com/langroid/langroid/blob/main/.env-template .env

# Edit the .env file with your favorite editor (here nano), 
# and add API keys as explained above
nano .env

# launch the container
docker run -it  -v ./.env:/.env langroid/langroid

# Use this command to run any of the scripts in the `examples` directory
python examples/<Path/To/Example.py> 
``` 



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



Click to expand any of the code examples below.
All of these can be run in a Colab notebook:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langroid/langroid/blob/main/examples/langroid_quick_examples.ipynb)

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
from langroid.vector_store.qdrantdb import QdrantDBConfig
config = DocChatAgentConfig(
  doc_paths = [
    "https://en.wikipedia.org/wiki/Language_model",
    "https://en.wikipedia.org/wiki/N-gram_language_model",
    "/path/to/my/notes-on-language-models.txt",
  ]
  llm = OpenAIGPTConfig(
    chat_model=OpenAIChatModel.GPT4,
  ),
  vecdb=QdrantDBConfig()
)
```

Then instantiate the `DocChatAgent` (this ingests the docs into the vector-store):

```python
agent = DocChatAgent(config)
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

<details>
<summary><b> :fire: Chat with tabular data (file paths, URLs, dataframes) </b></summary>

Using Langroid you can set up a `TableChatAgent` with a dataset (file path, URL or dataframe),
and query it. The Agent's LLM generates Pandas code to answer the query, 
via function-calling (or tool/plugin), and the Agent's function-handling method
executes the code and returns the answer.

Here is how you can do this:

```python
from langroid.agent.special.table_chat_agent import TableChatAgent, TableChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
```

Set up a `TableChatAgent` for a data file, URL or dataframe
(Ensure the data table has a header row; the delimiter/separator is auto-detected):
```python
dataset =  "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
# or dataset = "/path/to/my/data.csv"
# or dataset = pd.read_csv("/path/to/my/data.csv")
agent = TableChatAgent(
    config=TableChatAgentConfig(
        data=dataset,  
        llm=OpenAIGPTConfig(
            chat_model=OpenAIChatModel.GPT4,
        ),
    )
)
```
Set up a task, and ask one-off questions like this: 

```python
task = Task(
  agent, 
  name = "DataAssistant",
  default_human_response="", # to avoid waiting for user input
)
result = task.run(
  "What is the average alcohol content of wines with a quality rating above 7?",
  turns=2 # return after user question, LLM fun-call/tool response, Agent code-exec result
) 
print(result.content)
```
Or alternatively, set up a task and run it in an interactive loop with the user:

```python
task = Task(agent, name="DataAssistant")
task.run()
``` 

For a full working example see the 
[`table_chat.py`](https://github.com/langroid/langroid-examples/tree/main/examples/data-qa/table_chat.py)
script in the `langroid-examples` repo.


</details>

---

# :heart: Thank you to our [supporters](https://github.com/langroid/langroid/stargazers)

If you like this project, please give it a star ⭐ and 📢 spread the word in your network or social media:

[![Share on Twitter](https://img.shields.io/twitter/url?style=social&url=https://github.com/langroid/langroid)](https://twitter.com/intent/tweet?text=Langroid%20is%20a%20powerful,%20elegant%20new%20framework%20to%20easily%20build%20%23LLM%20applications.%20You%20set%20up%20LLM-powered%20Agents%20with%20vector-stores,%20assign%20tasks,%20and%20have%20them%20collaboratively%20solve%20problems%20via%20message-transformations.%20https://github.com/langroid/langroid)
[![Share on LinkedIn](https://img.shields.io/badge/Share%20on-LinkedIn-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https://github.com/langroid/langroid&title=Langroid:%20A%20Powerful,%20Elegant%20Framework&summary=Langroid%20is%20a%20powerful,%20elegant%20new%20framework%20to%20easily%20build%20%23LLM%20applications.%20You%20set%20up%20LLM-powered%20Agents%20with%20vector-stores,%20assign%20tasks,%20and%20have%20them%20collaboratively%20solve%20problems%20via%20message-transformations.)
[![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Flangroid%2Flangroid&t=Harness%20LLMs%20with%20Multi-Agent%20Programming)
[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-blue)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Flangroid%2Flangroid&title=Harness%20LLMs%20with%20Multi-Agent%20Programming)




Your support will help build Langroid's momentum and community.




# Langroid Co-Founders

- [Prasad Chalasani](https://www.linkedin.com/in/pchalasani/) (IIT BTech/CS, CMU PhD/ML; Independent ML Consultant)
- [Somesh Jha](https://www.linkedin.com/in/somesh-jha-80208015/) (IIT BTech/CS, CMU PhD/CS; Professor of CS, U Wisc at Madison)



