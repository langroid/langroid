# Options for accessing LLMs

> This is a work-in-progress document. It will be updated frequently.

The variety of ways to access the power of Large Language Models (LLMs) is growing 
rapidly, and there are a bewildering array of options. This document is an attempt to 
categorize and describe some of the most popular and useful ways to access LLMs,
via these 2x2x2  combinations:

- Websites (non-programmatic) or APIs (programmatic)
- Open-source or Proprietary 
- Chat-based interface or integrated assistive tools.

We will go into some of these combinations below. More will be added over time.

## Chat-based Web (non-API) access to Proprietary LLMs


This is best for *non-programmatic* use of LLMs: you go to a website and 
interact with the LLM via a chat interface -- 
you write prompts and/or upload documents, and the LLM responds with plain text
or can create artifacts (e.g. reports, code,
charts, podcasts, etc) that you can then copy into your files, workflow or codebase.
They typically allow you to upload text-based documents of various types, and some let you upload images, screen-shots, etc and ask questions about them.

Most of them are capable of doing *internet search* to inform their responses.


!!! note "Chat Interface vs Integrated Tools"
    Note that when using a chat-based interaction, you have to copy various artifacts
    from the web-site into another place, like your code editor, document, etc.
    AI-integrated tools relieve you of this burden by bringing the LLM power into 
    your workflow directly. More on this in a later section.

      
**Pre-requisites:** 

- *Computer*: Besides having a modern web browser (Chrome, Firefox, etc) and internet
access, there are no other special requirements, since the LLM is 
running on a remote server.
- *Coding knowledge*: Where (typically Python) code is produced, you will get best results
if you are conversant with Python so that you can understand and modify the code as
needed. In this category you do not need to know how to interact with an LLM API via code.

Here are some popular options in this category:

### OpenAI ChatGPT

Free access at [https://chatgpt.com/](https://chatgpt.com/)

With a ChatGPT-Plus monthly subscription ($20/month), you get additional features like:

- access to more powerful models
- access to [OpenAI canvas](https://help.openai.com/en/articles/9930697-what-is-the-canvas-feature-in-chatgpt-and-how-do-i-use-it) - this offers a richer interface than just a chat window, e.g. it automatically creates windows for code snippets, and shows results of running code
(e.g. output, charts etc).

Typical use: Since there is fixed monthly subscription (i.e. not metered by amount of 
usage), this is a cost-effective way to non-programmatically 
access a top LLM such as `GPT-4o` or `o1` 
(so-called "reasoning/thinking" models). Note however that there are limits on how many
queries you can make within a certain time period, but usually the limit is fairly
generous. 

What you can create, besides text-based artifacts:

- produce Python (or other language) code which you can copy/paste into notebooks or files
- SQL queries that you can copy/paste into a database tool
- Markdown-based tables
- You can't get diagrams, but you can get *code for diagrams*, 
e.g. python code for plots, [mermaid](https://github.com/mermaid-js/mermaid) code for flowcharts.
- images in some cases.

### OpenAI Custom GPTs (simply known as "GPTs")

[https://chatgpt.com/gpts/editor](https://chatgpt.com/gpts/editor)

Here you can conversationally interact with a "GPT Builder" that will 
create a version of ChatGPT
that is *customized* to your needs, i.e. with necessary background instructions,
context, and/or documents. 
The end result is a specialized GPT that you can then use for your specific
purpose and share with others (all of this is non-programmatic). 

E.g. [here](https://chatgpt.com/share/67153a4f-ea2c-8003-a6d3-cbc2412d78e5) is a "Knowledge Graph Builder" GPT

!!! note "Private GPTs requires an OpenAI Team Account"
    To share a custom GPT within a private group, you need an OpenAI Team account,
    see pricing [here](https://openai.com/chatgpt/pricing). Without a Team account,
    any shared GPT is public and can be accessed by anyone.


### Anthropic/Claude

[https://claude.ai](https://claude.ai)

The Claude basic web-based interface is similar to OpenAI ChatGPT, powered by 
Anthropic's proprietary LLMs. 
Anthropic's equivalent of ChatGPT-Plus is called "Claude Pro", which is also 
a $20/month subscription, giving you access to advanced models 
(e.g. `Claude-3.5-Sonnet`) and features.

Anthropic's equivalent of Custom GPTs is called 
[Projects](https://www.anthropic.com/news/projects), 
where you can create
an  LLM-powered interface that is augmented with your custom context and data.

Whichever product you are using, the interface auto-creates **artifacts** as needed --
these are stand-alone documents (code, text, images, web-pages, etc) 
that you may want to copy and paste into your own codebase, documents, etc.
For example you can prompt Claude to create full working interactive applications,
and copy the code, polish it and deploy it for others to use. See examples [here](https://simonwillison.net/2024/Oct/21/claude-artifacts/).

### Microsoft Copilot Lab

!!! note
    Microsoft's "Copilot" is an overloaded term that can refer to many different 
    AI-powered tools. Here we are referring to the one that is a collaboration between
    Microsoft and OpenAI, and is based on OpenAI's GPT-4o LLM, and powered by 
    Bing's search engine.

Accessible via [https://copilot.cloud.microsoft.com/](https://copilot.cloud.microsoft.com/)

The basic capabilities are similar to OpenAI's and Anthropic's offerings, but
come with so-called "enterprise grade" security and privacy features,
which purportedly make it suitable for use in educational and corporate settings.
Read more on what you can do with Copilot Lab [here](https://www.microsoft.com/en-us/microsoft-copilot/learn/?form=MA13FV).

Like the other proprietary offerings, Copilot can:

- perform internet search to inform its responses
- generate/run code and show results including charts

### Google Gemini

Accessible at [gemini.google.com](https://gemini.google.com).


## AI-powered productivity tools

These tools "bring the AI to your workflow", which is a massive productivity boost,
compared to repeatedly context-switching, e.g. copying/pasting between a chat-based AI web-app and your workflow.

- [**Cursor**](https://www.cursor.com/): AI Editor/Integrated Dev Environment (IDE). This is a fork of VSCode.
- [**Zed**](https://zed.dev/): built in Rust; can be customized to use Jetbrains/PyCharm keyboard shortcuts.
- [**Google Colab Notebooks with Gemini**](https://colab.research.google.com).
- [**Google NotebookLM**](https://notebooklm.google.com/): allows you to upload a set of text-based documents, 
  and create artifacts such as study guide, FAQ, summary, podcasts, etc.

    
## APIs for Proprietary LLMs

Using an API key allows *programmatic* access to the LLMs, meaning you can make
invocations to the LLM from within your own code, and receive back the results.
This is useful for building applications involving more complex workflows where LLMs
are used within a larger codebase, to access "intelligence" as needed.

E.g. suppose you are writing code that handles queries from a user, and you want to 
classify the user's _intent_ into one of 3 types: Information, or Action or Done.
Pre-LLMs, you would have had to write a bunch of rules or train a custom 
"intent classifier" that maps, for example:

- "What is the weather in Pittsburgh?" -> Information
- "Set a timer for 10 minutes" -> Action
- "Ok I have no more questions∞" -> Done

But using an LLM API, this is almost trivially easy - you instruct the LLM it should
classify the intent into one of these 3 types, and send the user query to the LLM,
and receive back the intent. 
(You can use Tools to make this robust, but that is outside the scope of this document.)

The most popular proprietary LLMs available via API are from OpenAI (or via  its
partner Microsoft), Anthropic, and Google:

- [OpenAI](https://platform.openai.com/docs/api-reference/introduction), to interact with `GPT-4o` family of models, and the `o1` family of "thinking/reasoning" models.
- [Anthropic](https://docs.anthropic.com/en/home) to use the `Claude` series of models.
- [Google](https://ai.google.dev/gemini-api/docs) to use the `Gemini` family of models.

These LLM providers are home to some of the most powerful LLMs available today,
specifically OpenAI's `GPT-4o` and Anthropic's `Claude-3.5-Sonnet`, and Google's `Gemini 1.5 Pro` (as of Oct 2024).

**Billing:** Unlike the fixed monthly subscriptions of ChatGPT, Claude and others, 
LLM usage via API is typically billed by *token usage*, i.e. you pay for the total
number of input and output "tokens" (a slightly technical term, but think of it as
a word for now).

Using an LLM API involves these steps:

- create an account on the provider's website as a "developer" or organization,
- get an API key,
- use the API key in your code to make requests to the LLM. 


**Prerequisites**:

- *Computer:* again, since the API is served over the internet, there are no special
  requirements for your computer.
- *Programming skills:* Using an LLM API involves either:
    - directly making REST API calls from your code, or 
    - use a scaffolding library (like [Langroid](https://github.com/langroid/langroid)) that abstracts away the details of the 
      API calls.
  
    In either case, you must be highly proficient in (Python) programming 
  to use this option.

## Web-interfaces to Open LLMs

!!! note  "Open LLMs"
    These are LLMs that have been publicly released, i.e. their parameters ("weights") 
    are publicly available -- we refer to these as *open-weight* LLMs. If in addition, the
    training datasets, and data-preprocessing and training code are also available, we would
    call these *open-source* LLMs. But lately there is a looser usage of the term "open-source",referring to just the weights being available. For our purposes we will just refer all of these models as **Open LLMs**.

There are many options here, but some popular ones are below. Note that some of these
are front-ends that allow you to interact with not only Open LLMs but also 
proprietary LLM APIs.

- [LMStudio](https://lmstudio.ai/)
- [OpenWebUI](https://github.com/open-webui/open-webui)
- [Msty](https://msty.app/)
- [AnythingLLM](https://anythingllm.com/)
- [LibreChat](https://www.librechat.ai/)


## API Access to Open LLMs

This is a good option if you are fairly proficient in (Python) coding. There are in 
fact two possibilities here:

- The LLM is hosted remotely, and you make REST API calls to the remote server. This
  is a good option when you want to run large LLMs and you don't have the resources (GPU and memory) to run them locally.
    - [groq](https://groq.com/) amazingly it is free, and you can run `llama-3.1-70b`
    - [cerebras](https://cerebras.ai/)
    - [open-router](https://openrouter.ai/)
    - [AI/ML API](https://aimlapi.com/app/?utm_source=langroid&utm_medium=github&utm_campaign=integration)
- The LLM is running on your computer. This is a good option if your machine has sufficient RAM to accommodate the LLM you are trying to run, and if you are 
concerned about data privacy. The most user-friendly option is [Ollama](https://github.com/ollama/ollama); see more below.

Note that all of the above options provide an **OpenAI-Compatible API** to interact
with the LLM, which is a huge convenience: you can write code to interact with OpenAI's
LLMs (e.g. `GPT4o` etc) and then easily switch to one of the above options, typically
by changing a simple config (see the respective websites for instructions).

Of course, directly working with the raw LLM API quickly becomes tedious. This is where
a scaffolding library like [langroid](https://github.com/langroid/langroid) comes in
very handy - it abstracts away the details of the API calls, and provides a simple
programmatic interface to the LLM, and higher-level abstractions like 
Agents, Tasks, etc. Working with such a library is going to be far more productive
than directly working with the raw API. Below are instructions on how to use langroid
with some the above Open/Local LLM options.

See [here](https://langroid.github.io/langroid/tutorials/local-llm-setup/) for 
a guide to using Langroid with Open LLMs.


