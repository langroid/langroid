<div align="center">
  <img src="docs/assets/langroid-card-lambda-ossem-rust-1200-630.png" alt="Logo" 
        width="400" align="center">
</div>

<div align="center">

[![PyPI - Version](https://img.shields.io/pypi/v/langroid)](https://pypi.org/project/langroid/)
[![Pytest](https://github.com/langroid/langroid/actions/workflows/pytest.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/langroid/langroid/branch/main/graph/badge.svg?token=H94BX5F0TE)](https://codecov.io/gh/langroid/langroid)
[![Multi-Architecture DockerHub](https://github.com/langroid/langroid/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/langroid/langroid/actions/workflows/docker-publish.yml)

[![Static Badge](https://img.shields.io/badge/Documentation-blue?link=https%3A%2F%2Flangroid.github.io%2Flangroid%2F&link=https%3A%2F%2Flangroid.github.io%2Flangroid%2F)](https://langroid.github.io/langroid)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langroid/langroid/blob/main/examples/Langroid_quick_start.ipynb)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=flat&logo=discord&logoColor=white)](https://discord.gg/ZU36McDgDs)
[![Substack](https://img.shields.io/badge/Substack-%23006f5c.svg?style=flat&logo=substack&logoColor=FF6719)](https://langroid.substack.com/p/langroid-harness-llms-with-multi-agent-programming)
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
Python framework to easily build LLM-powered applications, from ex-CMU and UW-Madison researchers. 
You set up Agents, equip them with optional components (LLM, 
vector-store and tools/functions), assign them tasks, and have them 
collaboratively solve a problem by exchanging messages. 
This Multi-Agent paradigm is inspired by the
[Actor Framework](https://en.wikipedia.org/wiki/Actor_model)
(but you do not need to know anything about this!). 

`Langroid` is a fresh take on LLM app-development, where considerable thought has gone 
into simplifying the developer experience; it does not use `Langchain`.

:fire: See this [Intro to Langroid](https://lancedb.substack.com/p/langoid-multi-agent-programming-framework)
blog post from the LanceDB team


We welcome contributions -- See the [contributions](./CONTRIBUTING.md) document
for ideas on what to contribute.

Are you building LLM Applications, or want help with Langroid for your company, 
or want to prioritize Langroid features for your company use-cases? 
[Prasad Chalasani](https://www.linkedin.com/in/pchalasani/) is available for consulting
(advisory/development): pchalasani at gmail dot com.

Sponsorship is also accepted via [GitHub Sponsors](https://github.com/sponsors/langroid)

**Questions, Feedback, Ideas? Join us on [Discord](https://discord.gg/ZU36McDgDs)!**

# Quick glimpse of coding with Langroid
This is just a teaser; there's much more, like function-calling/tools, 
Multi-Agent Collaboration, Structured Information Extraction, DocChatAgent 
(RAG), SQLChatAgent, non-OpenAI local/remote LLMs, etc. Scroll down or see docs for more.
See the Langroid Quick-Start [Colab](https://colab.research.google.com/github/langroid/langroid/blob/main/examples/Langroid_quick_start.ipynb)
that builds up to a 2-agent information-extraction example using the OpenAI ChatCompletion API. 
See also this [version](https://colab.research.google.com/drive/190Tk7t4AdY1P9F_NlZ33-YEoGnHweQQ0) that uses the OpenAI Assistants API instead.

:fire: just released! [Example](https://github.com/langroid/langroid-examples/blob/main/examples/docqa/chat-multi-extract-local.py) 
script showing how you can use Langroid multi-agents and tools
to extract structured information from a document using **only a local LLM**
(Mistral-7b-instruct-v0.2).

```python
import langroid as lr
import langroid.language_models as lm

# set up LLM
llm_cfg = lm.OpenAIGPTConfig( # or OpenAIAssistant to use Assistant API 
  # any model served via an OpenAI-compatible API
  chat_model=lm.OpenAIChatModel.GPT4_TURBO, # or, e.g., "ollama/mistral"
)
# use LLM directly
mdl = lm.OpenAIGPT(llm_cfg)
response = mdl.chat("What is the capital of Ontario?", max_tokens=10)

# use LLM in an Agent
agent_cfg = lr.ChatAgentConfig(llm=llm_cfg)
agent = lr.ChatAgent(agent_cfg)
agent.llm_response("What is the capital of China?") 
response = agent.llm_response("And India?") # maintains conversation state 

# wrap Agent in a Task to run interactive loop with user (or other agents)
task = lr.Task(agent, name="Bot", system_message="You are a helpful assistant")
task.run("Hello") # kick off with user saying "Hello"

# 2-Agent chat loop: Teacher Agent asks questions to Student Agent
teacher_agent = lr.ChatAgent(agent_cfg)
teacher_task = lr.Task(
  teacher_agent, name="Teacher",
  system_message="""
    Ask your student concise numbers questions, and give feedback. 
    Start with a question.
    """
)
student_agent = lr.ChatAgent(agent_cfg)
student_task = lr.Task(
  student_agent, name="Student",
  system_message="Concisely answer the teacher's questions.",
  single_round=True,
)

teacher_task.add_sub_task(student_task)
teacher_task.run()
```

# :fire: Updates/Releases

<details>
<summary> <b>Click to expand</b></summary>

- **May 2024:** 
  - `gpt-4o` is now the default LLM throughout; Update tests and examples to work 
    with this LLM; use tokenizer corresponding to the LLM.
  - `gemini 1.5 pro` support via `litellm`
  - `QdrantDB:` update to support learned sparse embeddings.
- **Apr 2024:**
  - **0.1.236**: Support for open LLMs hosted on Groq, e.g. specify 
    `chat_model="groq/llama3-8b-8192"`.
      See [tutorial](https://langroid.github.io/langroid/tutorials/local-llm-setup/).
  - **0.1.235**: `Task.run(), Task.run_async(), run_batch_tasks` have `max_cost` 
    and `max_tokens` params to exit when tokens or cost exceed a limit. The result 
    `ChatDocument.metadata` now includes a `status` field which is a code indicating a 
     task completion reason code. Also `task.run()` etc can be invoked with an explicit
     `session_id` field which is used as a key to look up various settings in Redis cache.
    Currently only used to look up "kill status" - this allows killing a running task, either by `task.kill()`
    or by the classmethod `Task.kill_session(session_id)`.
    For example usage, see the `test_task_kill` in [tests/main/test_task.py](https://github.com/langroid/langroid/blob/main/tests/main/test_task.py)
  
- **Mar 2024:**
  - **0.1.216:** Improvements to allow concurrent runs of `DocChatAgent`, see the
    [`test_doc_chat_agent.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_doc_chat_agent.py)
    in particular the `test_doc_chat_batch()`;
    New task run utility: [`run_batch_task_gen`](https://github.com/langroid/langroid/blob/main/langroid/agent/batch.py) 
    where a task generator can be specified, to generate one task per input. 
  - **0.1.212:** ImagePdfParser: support for extracting text from image-based PDFs.
    (this means `DocChatAgent` will now work with image-pdfs).
  - **0.1.194 - 0.1.211:** Misc fixes, improvements, and features:
    - Big enhancement in RAG performance (mainly, recall) due to a [fix in Relevance 
      Extractor](https://github.com/langroid/langroid/releases/tag/0.1.209)
    - `DocChatAgent` [context-window fixes](https://github.com/langroid/langroid/releases/tag/0.1.208)
    - Anthropic/Claude3 support via Litellm
    - `URLLoader`: detect file time from header when URL doesn't end with a 
      recognizable suffix like `.pdf`, `.docx`, etc.
    - Misc lancedb integration fixes
    - Auto-select embedding config based on whether `sentence_transformer` module is available.
    - Slim down dependencies, make some heavy ones optional, e.g. `unstructured`, 
      `haystack`, `chromadb`, `mkdocs`, `huggingface-hub`, `sentence-transformers`.
    - Easier top-level imports from `import langroid as lr`
    - Improve JSON detection, esp from weak LLMs
- **Feb 2024:** 
  - **0.1.193:** Support local LLMs using Ollama's new OpenAI-Compatible server: 
     simply specify `chat_model="ollama/mistral"`. See [release notes](https://github.com/langroid/langroid/releases/tag/0.1.193).
  - **0.1.183:** Added Chainlit support via [callbacks](https://github.com/langroid/langroid/blob/main/langroid/agent/callbacks/chainlit.py). 
   See [examples](https://github.com/langroid/langroid/tree/main/examples/chainlit).
- **Jan 2024:**
  - **0.1.175** 
    - [Neo4jChatAgent](https://github.com/langroid/langroid/tree/main/langroid/agent/special/neo4j) to chat with a neo4j knowledge-graph.
      (Thanks to [Mohannad](https://github.com/Mohannadcse)!). The agent uses tools to query the Neo4j schema and translate user queries to Cypher queries,
      and the tool handler executes these queries, returning them to the LLM to compose
      a natural language response (analogous to how `SQLChatAgent` works).
      See example [script](https://github.com/langroid/langroid/tree/main/examples/kg-chat) using this Agent to answer questions about Python pkg dependencies.
    - Support for `.doc` file parsing (in addition to `.docx`)
    - Specify optional [`formatter` param](https://github.com/langroid/langroid/releases/tag/0.1.171) 
      in `OpenAIGPTConfig` to ensure accurate chat formatting for local LLMs. 
  - **[0.1.157](https://github.com/langroid/langroid/releases/tag/0.1.157):** `DocChatAgentConfig` 
     has a new param: `add_fields_to_content`, to specify additional document fields to insert into 
     the main `content` field, to help improve retrieval.
  - **[0.1.156](https://github.com/langroid/langroid/releases/tag/0.1.156):** New Task control signals
     PASS_TO, SEND_TO; VectorStore: Compute Pandas expression on documents; LanceRAGTaskCreator creates 3-agent RAG system with Query Planner, Critic and RAG Agent.
- **Dec 2023:**
  - **0.1.154:** (For details see release notes of [0.1.149](https://github.com/langroid/langroid/releases/tag/0.1.149)
      and [0.1.154](https://github.com/langroid/langroid/releases/tag/0.1.154)). 
    - `DocChatAgent`: Ingest Pandas dataframes and filtering.
    - `LanceDocChatAgent` leverages `LanceDB` vector-db for efficient vector search
     and full-text search and filtering.
    - Improved task and multi-agent control mechanisms
    - `LanceRAGTaskCreator` to create a 2-agent system consisting of a `LanceFilterAgent` that
      decides a filter and rephrase query to send to a RAG agent.
  - **[0.1.141](https://github.com/langroid/langroid/releases/tag/0.1.141):**
    API Simplifications to reduce boilerplate:
    auto-select an available OpenAI model (preferring gpt-4-turbo), simplifies defaults.
    Simpler `Task` initialization with default `ChatAgent`.
- **Nov 2023:**
  - **[0.1.126](https://github.com/langroid/langroid/releases/tag/0.1.126):**
     OpenAIAssistant agent: Caching Support. 
  - **0.1.117:** Support for OpenAI Assistant API tools: Function-calling, 
    Code-intepreter, and Retriever (RAG), file uploads. These work seamlessly 
    with Langroid's task-orchestration.
    Until docs are ready, it's best to see these usage examples:
    
    - **Tests:**
      - [test_openai_assistant.py](https://github.com/langroid/langroid/blob/main/tests/main/test_openai_assistant.py)
      - [test_openai_assistant_async.py](https://github.com/langroid/langroid/blob/main/tests/main/test_openai_assistant_async.py)

    - **Example scripts:**
      - [The most basic chat app](https://github.com/langroid/langroid/blob/main/examples/basic/oai-asst-chat.py)
      - [Chat with code interpreter](https://github.com/langroid/langroid/blob/main/examples/basic/oai-code-chat.py)
      - [Chat with retrieval (RAG)](https://github.com/langroid/langroid/blob/main/examples/docqa/oai-retrieval-assistant.py)
      - [2-agent RAG chat](https://github.com/langroid/langroid/blob/main/examples/docqa/oai-retrieval-2.py)
  - **0.1.112:** [`OpenAIAssistant`](https://github.com/langroid/langroid/blob/main/langroid/agent/openai_assistant.py) is a subclass of `ChatAgent` that 
    leverages the new OpenAI Assistant API. It can be used as a drop-in 
    replacement for `ChatAgent`, and relies on the Assistant API to
    maintain conversation state, and leverages persistent threads and 
    assistants to reconnect to them if needed. Examples: 
    [`test_openai_assistant.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_openai_assistant.py),
    [`test_openai_assistant_async.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_openai_assistant_async.py)
  - **0.1.111:** Support latest OpenAI model: `GPT4_TURBO`
(see [test_llm.py](tests/main/test_llm.py) for example usage)
  - **0.1.110:** Upgrade from OpenAI v0.x to v1.1.1 (in preparation for 
    Assistants API and more); (`litellm` temporarily disabled due to OpenAI 
    version conflict).
- **Oct 2023:**
  - **0.1.107:** `DocChatAgent` re-rankers: `rank_with_diversity`, `rank_to_periphery` (lost in middle).
  - **0.1.102:** `DocChatAgentConfig.n_neighbor_chunks > 0` allows returning context chunks around match.
  - **0.1.101:** `DocChatAgent` uses `RelevanceExtractorAgent` to have 
    the LLM extract relevant portions of a chunk using 
    sentence-numbering, resulting in huge speed up and cost reduction 
    compared to the naive "sentence-parroting" approach (writing out full 
    sentences out relevant whole sentences) which `LangChain` uses in their 
    `LLMChainExtractor`.
  - **0.1.100:** API update: all of Langroid is accessible with a single import, i.e. `import langroid as lr`. See the [documentation]("https://langroid.github.io/langroid/") for usage.
  - **0.1.99:** Convenience batch functions to run tasks, agent methods on a list of inputs concurrently in async mode. See examples in [test_batch.py](https://github.com/langroid/langroid/blob/main/tests/main/test_batch.py).
  - **0.1.95:** Added support for [Momento Serverless Vector Index](https://docs.momentohq.com/vector-index)
  - **0.1.94:** Added support for [LanceDB](https://lancedb.github.io/lancedb/) vector-store -- allows vector, Full-text, SQL search.
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
  - **Colab notebook** to try the quick-start examples: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langroid/langroid/blob/main/examples/Langroid_quick_start.ipynb) 
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
(See [this script](https://github.com/langroid/langroid-examples/blob/main/examples/docqa/chat-multi-extract-local.py)
for a version with the same functionality using a local Mistral-7b model.)
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
(For a more up-to-date list see the 
[release](https://github.com/langroid/langroid?tab=readme-ov-file#fire-updatesreleases) 
section above)
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
- **LLM Support**: Langroid supports OpenAI LLMs as well as LLMs from hundreds of 
providers (local/open or remote/commercial) via proxy libraries and local model servers
such as [LiteLLM](https://docs.litellm.ai/docs/providers) that in effect mimic the OpenAI API. 
- **Caching of LLM responses:** Langroid supports [Redis](https://redis.com/try-free/) and 
  [Momento](https://www.gomomento.com/) to cache LLM responses.
- **Vector-stores**: [LanceDB](https://github.com/lancedb/lancedb), [Qdrant](https://qdrant.tech/), [Chroma](https://www.trychroma.com/) are currently supported.
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
If using `zsh` (or similar shells), you may need to escape the square brackets, e.g.:
```
pip install langroid\[hf-embeddings\]
```
or use quotes:
```
pip install "langroid[hf-embeddings]"
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
Your `.env` file should look like this (the organization is optional 
but may be required in some scenarios).
```bash
OPENAI_API_KEY=your-key-here-without-quotes
OPENAI_ORGANIZATION=optionally-your-organization-id
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
  The default vector store in our RAG agent (`DocChatAgent`) is LanceDB which uses file storage,
  and you do not need to set up any environment variables for that.
  Alternatively [Chroma](https://docs.trychroma.com/) is also currently supported. 
  We use the local-storage version of Chroma, so there is no need for an API key.
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

- `AZURE_OPENAI_API_KEY`, from the value of `API_KEY`
- `AZURE_OPENAI_API_BASE` from the value of `ENDPOINT`, typically looks like `https://your.domain.azure.com`.
- For `AZURE_OPENAI_API_VERSION`, you can use the default value in `.env-template`, and latest version can be found [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new#azure-openai-chat-completion-general-availability-ga)
- `AZURE_OPENAI_DEPLOYMENT_NAME` is the name of the deployed model, which is defined by the user during the model setup 
- `AZURE_OPENAI_MODEL_NAME` Azure OpenAI allows specific model names when you select the model for your deployment. You need to put precisly the exact model name that was selected. For example, GPT-3.5 (should be `gpt-35-turbo-16k` or `gpt-35-turbo`) or GPT-4 (should be `gpt-4-32k` or `gpt-4`).
- `AZURE_OPENAI_MODEL_VERSION` is required if `AZURE_OPENAI_MODEL_NAME = gpt=4`, which will assist Langroid to determine the cost of the model  
</details>

---

# :whale: Docker Instructions

We provide a containerized version of the [`langroid-examples`](https://github.com/langroid/langroid-examples) 
repository via this [Docker Image](https://hub.docker.com/r/langroid/langroid).
All you need to do is set up environment variables in the `.env` file.
Please follow these steps to setup the container:

```bash
# get the .env file template from `langroid` repo
wget -O .env https://raw.githubusercontent.com/langroid/langroid/main/.env-template

# Edit the .env file with your favorite editor (here nano), and remove any un-used settings. E.g. there are "dummy" values like "your-redis-port" etc -- if you are not using them, you MUST remove them.
nano .env

# launch the container
docker run -it --rm  -v ./.env:/langroid/.env langroid/langroid

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
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langroid/langroid/blob/main/examples/Langroid_quick_start.ipynb)

<details>
<summary> <b> Direct interaction with OpenAI LLM </b> </summary>

```python
import langroid.language_models as lm

mdl = lm.OpenAIGPT()

messages = [
  lm.LLMMessage(content="You are a helpful assistant",  role=lm.Role.SYSTEM), 
  lm.LLMMessage(content="What is the capital of Ontario?",  role=lm.Role.USER),
]

response = mdl.chat(messages, max_tokens=200)
print(response.message)
```
</details>

<details>
<summary> <b> Interaction with non-OpenAI LLM (local or remote) </b> </summary>
Local model: if model is served at `http://localhost:8000`:

```python
cfg = lm.OpenAIGPTConfig(
  chat_model="local/localhost:8000", 
  chat_context_length=4096
)
mdl = lm.OpenAIGPT(cfg)
# now interact with it as above, or create an Agent + Task as shown below.
```

If the model is [supported by `liteLLM`](https://docs.litellm.ai/docs/providers), 
then no need to launch the proxy server.
Just set the `chat_model` param above to `litellm/[provider]/[model]`, e.g. 
`litellm/anthropic/claude-instant-1` and use the config object as above.
Note that to use `litellm` you need to install langroid with the `litellm` extra:
`poetry install -E litellm` or `pip install langroid[litellm]`.
For remote models, you will typically need to set API Keys etc as environment variables.
You can set those based on the LiteLLM docs. 
If any required environment variables are missing, Langroid gives a helpful error
message indicating which ones are needed.
Note that to use `langroid` with `litellm` you need to install the `litellm` 
extra, i.e. either `pip install langroid[litellm]` in your virtual env,
or if you are developing within the `langroid` repo, 
`poetry install -E litellm`.
```bash
pip install langroid[litellm]
```
</details>

<details>
<summary> <b> Define an agent, set up a task, and run it </b> </summary>

```python
import langroid as lr

agent = lr.ChatAgent()

# get response from agent's LLM, and put this in an interactive loop...
# answer = agent.llm_response("What is the capital of Ontario?")
  # ... OR instead, set up a task (which has a built-in loop) and run it
task = lr.Task(agent, name="Bot") 
task.run() # ... a loop seeking response from LLM or User at each turn
```
</details>

<details>
<summary><b> Three communicating agents </b></summary>

A toy numbers game, where when given a number `n`:
- `repeater_task`'s LLM simply returns `n`,
- `even_task`'s LLM returns `n/2` if `n` is even, else says "DO-NOT-KNOW"
- `odd_task`'s LLM returns `3*n+1` if `n` is odd, else says "DO-NOT-KNOW"

Each of these `Task`s automatically configures a default `ChatAgent`.

```python
import langroid as lr
from langroid.utils.constants import NO_ANSWER

repeater_task = lr.Task(
    name = "Repeater",
    system_message="""
    Your job is to repeat whatever number you receive.
    """,
    llm_delegate=True, # LLM takes charge of task
    single_round=False, 
)

even_task = lr.Task(
    name = "EvenHandler",
    system_message=f"""
    You will be given a number. 
    If it is even, divide by 2 and say the result, nothing else.
    If it is odd, say {NO_ANSWER}
    """,
    single_round=True,  # task done after 1 step() with valid response
)

odd_task = lr.Task(
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
tool syntax, Langroid sends the Pydantic validation error (suitably sanitized) 
to the LLM so it can fix it!

Simple example: Say the agent has a secret list of numbers, 
and we want the LLM to find the smallest number in the list. 
We want to give the LLM a `probe` tool/function which takes a
single number `n` as argument. The tool handler method in the agent
returns how many numbers in its list are at most `n`.

First define the tool using Langroid's `ToolMessage` class:


```python
import langroid as lr

class ProbeTool(lr.agent.ToolMessage):
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
class SpyGameAgent(lr.ChatAgent):
  def __init__(self, config: lr.ChatAgentConfig):
    super().__init__(config)
    self.numbers = [3, 4, 8, 11, 15, 25, 40, 80, 90]

  def probe(self, msg: ProbeTool) -> str:
    # return how many numbers in self.numbers are less or equal to msg.number
    return str(len([n for n in self.numbers if n <= msg.number]))
```

We then instantiate the agent and enable it to use and respond to the tool:

```python
spy_game_agent = SpyGameAgent(
    lr.ChatAgentConfig(
        name="Spy",
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
import langroid as lr

class LeaseMessage(lr.agent.ToolMessage):
    request: str = "lease_info"
    purpose: str = """
        Collect information about a Commercial Lease.
        """
    terms: Lease
```

Then define a `LeaseExtractorAgent` with a method `lease_info` that handles this tool,
instantiate the agent, and enable it to use and respond to this tool:

```python
class LeaseExtractorAgent(lr.ChatAgent):
    def lease_info(self, message: LeaseMessage) -> str:
        print(
            f"""
        DONE! Successfully extracted Lease Info:
        {message.terms}
        """
        )
        return json.dumps(message.terms.dict())
    
lease_extractor_agent = LeaseExtractorAgent()
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
import langroid as lr
from langroid.agent.special import DocChatAgentConfig, DocChatAgent

config = DocChatAgentConfig(
  doc_paths = [
    "https://en.wikipedia.org/wiki/Language_model",
    "https://en.wikipedia.org/wiki/N-gram_language_model",
    "/path/to/my/notes-on-language-models.txt",
  ],
  vecdb=lr.vector_store.LanceDBConfig(),
)
```

Then instantiate the `DocChatAgent` (this ingests the docs into the vector-store):

```python
agent = DocChatAgent(config)
```
Then we can either ask the agent one-off questions,
```python
agent.llm_response("What is a language model?")
```
or wrap it in a `Task` and run an interactive loop with the user:
```python
task = lr.Task(agent)
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
import langroid as lr
from langroid.agent.special import TableChatAgent, TableChatAgentConfig
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
    )
)
```
Set up a task, and ask one-off questions like this: 

```python
task = lr.Task(
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
task = lr.Task(agent, name="DataAssistant")
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



