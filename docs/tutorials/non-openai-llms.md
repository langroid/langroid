# Using Langroid with Non-OpenAI LLMs

Langroid currently supports any LLM API that is compatible with the OpenAI API. 
This means two things:

- You can use Langroid with any locally-served LLM that _mimics_ the OpenAI API, 
    and the `openai` python client will continue to work, by simply changing the 
    `openai.api_base` to point to the local server 
    (see the [local-models](../blog/posts/local-llm.md) blog post for details.)
- You can use langroid with other (local or remote) LLMs, using an "adapter" or "proxy" library that _translates_ 
    between the OpenAI API to that LLM's API. The excellent [`LiteLLM`](https://github.com/berriai/litellm)
   is the best (and possibly only) such library, which was integrated into Langroid in version 0.1.84.

Below we outline the second approach, i.e. using `LiteLLM` to enable Langroid to work with any LLM 
supported by LiteLLM (over a 100 LLM providers).

## Switch to any LiteLLM-supported model, using environment variables

If you want to run an existing Langroid script with a different LLM that is supported by LiteLLM,
and don't want to change any code or write new code, you will want to follow this method.
Here are the steps, using AWS Bedrock's `claude-instant-v1` as an example:

- Look up the LiteLLM instructions for that model, e.g. [here](https://docs.litellm.ai/docs/providers/bedrock#required-environment-variables).
- On that page you will see that you need to set some environment variables. 
Set these in your `.env` file or explicitly at the command-line using 
  `export` or `setenv`. For example in the `.env` you will add these:

```bash
AWS_ACCESS_KEY_ID=<your-aws-access-key-id>
AWS_SECRET_ACCESS_KEY=<your-aws-secret-access-key>
AWS_REGION=<your-aws-region>
OPENAI_LITELLM=true # this tells Langroid to use LiteLLM
```

or explicitly assign these in your python script:

```python
import os
os.environ["AWS_ACCESS_KEY_ID"] = ""  # Access key
os.environ["AWS_SECRET_ACCESS_KEY"] = "" # Secret access key
os.environ["AWS_REGION_NAME"] = "" # us-east-1, us-east-2, us-west-1, us-west-2
os.environ["OPENAI_LITELLM"] = "true"
```

- On that same LiteLLM instruction page, you will see the model name needs to 
be specified. Set the required model name in your `.env` file or explicitly at the command-line using 
  `export` or `setenv`. For example in the `.env` you will add this:

```bash
OPENAI_CHAT_MODEL=bedrock/anthropic.claude-instant-v1
```
or explicitly assign this in your python script:

```python
os.environ["OPENAI_CHAT_MODEL"] = "bedrock/anthropic.claude-instant-v1"
```

You will now be able to use any existing Langroid script with this LLM
instead of the default GPT-4. Of course you will need to refine the prompts to get the 
results you want. 

## Finer control, by creating a subclass of `OpenAIGPTConfig`


If you are writing code in the Langroid framework and you want finer control on 
which LLM is being used (and perhaps allow different agents to use different LLMs),
in addition to setting the `AWS_*` environment variables above,
you will want to additionally do the following. We again use the AWS Bedrock `claud-instant-v1`
example here.

First create a subclass of `OpenAIGPTConfig`, with a suitable prefix, say 
"BEDROCK" (see below for why it is useful to have a special prefix).

```python
from langroid.language_models.openai_gpt import OpenAIGPTConfig
LiteLLMBedrockConfig = OpenAIGPTConfig.create(prefix="BEDROCK")
```

Next, instantiate an instance of this Config class:

```python
litellm_bedrock_config = LiteLLMBedrockConfig(
    chat_model="bedrock/anthropic.claude-instant-v1",
    litellm=True,
    chat_context_length=4096, 
)
```
Now you can use this LLM config instance to create a `ChatAgentConfig`, 
which you then use to instantiate a `ChatAgent`.

```python
from langroid.agent.base import ChatAgent, ChatAgentConfig
agent_config = ChatAgentConfig(
    llm=litellm_bedrock_config,
    name="bedrock-chat",
    vecdb=None, # or a suitable vector db config
)
agent = ChatAgent(agent_config)
```

Recall that we used a prefix `BEDROCK` in the `OpenAIGPTConfig` subclass we 
created above. This is useful because the `OpenAIGPTConfig` class is derived from 
Pydantic's `BaseSettings` class, which allows you to set environment variables
with a prefix. So suppose you want to switch your code to another model 
offered in AWS Bedrock. You can set the `BEDROCK_CHAT_MODEL` to this new model,
and it will be used without needing to change any of your code
(the model in your code will be used as a fallback if the environment variable is not set).

```bash
BEDROCK_CHAT_MODEL=bedrock/ai21.j2-ultra
BEDROCK_CHAT_CONTEXT_LENGTH=...
```

The environment prefix also helps when you want to have different agents use
different models, each with their own environment settings. If you create
subclasses of `OpenAIGPTConfig` with different prefixes, you can set the
environment variables for each of these models separately, and have all of these 
in the `.env` file without any conflicts.

For a working example, see [basic chat script](https://github.com/langroid/langroid/blob/main/examples/basic/chat.py)
where we apply this method to a locally-running model spun up using [ollama](https://github.com/jmorganca/ollama).





    