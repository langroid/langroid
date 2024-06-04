# Guide to examples in `langroid-examples` repo

!!! warning "Outdated"
    This guide is from Feb 2024; there have been numerous additional examples
    since then. We recommend you visit the `examples` folder in the core `langroid`
    repo for the most up-to-date examples. These examples are periodically copied
    over to the `examples` folder in the `langroid-examples` repo.

The [`langroid-examples`](https://github.com/langroid/langroid-examples) repo
contains several examples of using
the [Langroid](https://github.com/langroid/langroid) agent-oriented programming 
framework for LLM applications.
Below is a guide to the examples. First please ensure you follow the
installation instructions in the `langroid-examples` repo README.

**At minimum a GPT4-compatible OpenAI API key is required.** As currently set
up, many of the examples will _not_ work with a weaker model. Weaker models may
require more detailed or different prompting, and possibly a more iterative
approach with multiple agents to verify and retry, etc — this is on our roadmap.

All the example scripts are meant to be run on the command line.
In each script there is a description and sometimes instructions on how to run
the script.

NOTE: When you run any script, it pauses for “human” input at every step, and
depending on the context, you can either hit enter to continue, or in case there
is a question/response expected from the human, you can enter your question or
response and then hit enter.

### Basic Examples
- [`/examples/basic/chat.py`](https://github.com/langroid/langroid-examples/blob/main/examples/basic/chat.py) This is a basic chat application.

    - Illustrates Agent task loop.

- [`/examples/basic/autocorrect.py`](https://github.com/langroid/langroid-examples/blob/main/examples/basic/autocorrect.py) Chat with autocorrect: type fast and carelessly/lazily and 
the LLM will try its best to interpret what you want, and offer choices when confused.

    - Illustrates Agent task loop.

- [`/examples/basic/chat-search.py`](https://github.com/langroid/langroid-examples/blob/main/examples/basic/chat-search.py)  This uses a `GoogleSearchTool` function-call/tool to answer questions using a google web search if needed.
  Try asking questions about facts known after Sep 2021 (GPT4 training cutoff),
  like  `when was llama2 released`
  
    - Illustrates Agent + Tools/function-calling + web-search

- [`/examples/basic/chat-tree.py`](https://github.com/langroid/langroid-examples/blob/main/examples/basic/chat-tree.py) is a toy example of tree-structured multi-agent
  computation, see a detailed writeup [here.](https://langroid.github.io/langroid/examples/agent-tree/)
  
    - Illustrates multi-agent task collaboration, task delegation.

### Document-chat examples, or RAG (Retrieval Augmented Generation)

- [`/examples/docqa/chat.py`](https://github.com/langroid/langroid-examples/blob/main/examples/docqa/chat.py) is a document-chat application. Point it to local file,
  directory or web url, and ask questions
    - Illustrates basic RAG
- [`/examples/docqa/chat-search.py`](https://github.com/langroid/langroid-examples/blob/main/examples/docqa/chat-search.py): ask about anything and it will try to answer
  based on docs indexed in vector-db, otherwise it will do a Google search, and
  index the results in the vec-db for this and later answers.
    - Illustrates RAG + Function-calling/tools
- [`/examples/docqa/chat_multi.py`](https://github.com/langroid/langroid-examples/blob/main/examples/docqa/chat_multi.py):  — this is a 2-agent system that will summarize
  a large document with 5 bullet points: the first agent generates questions for
  the retrieval agent, and is done when it gathers 5 key points.
    - Illustrates 2-agent collaboration + RAG to summarize a document
- [`/examples/docqa/chat_multi_extract.py`](https://github.com/langroid/langroid-examples/blob/main/examples/docqa/chat_multi_extract.py):  — extracts structured info from a
  lease document: Main agent asks questions to a retrieval agent. 
    - Illustrates 2-agent collaboration, RAG, Function-calling/tools, Structured Information Extraction.

### Data-chat examples (tabular, SQL)

- [`/examples/data-qa/table_chat.py`](https://github.com/langroid/langroid-examples/blob/main/examples/data-qa/table_chat):  - point to a URL or local csv file and ask
  questions. The agent generates pandas code that is run within langroid.
    - Illustrates function-calling/tools and code-generation
- [`/examples/data-qa/sql-chat/sql_chat.py`](https://github.com/langroid/langroid-examples/blob/main/examples/data-qa/sql-chat/sql_chat.py):  — chat with a sql db — ask questions in
  English, it will generate sql code to answer them.
  See [tutorial here](https://langroid.github.io/langroid/tutorials/postgresql-agent/)
    - Illustrates function-calling/tools and code-generation

