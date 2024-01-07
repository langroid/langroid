"""
Basic chat example with a local LLM, e.g. after doing `ollama pull mistral`.
See full guide to using Langroid with non-OpenAI LLMs at
https://langroid.github.io/langroid/tutorials/non-openai-llms/

For some local-model scenarios (e.g. ollama) you must ensure you have installed the
`litellm` extra in your env, e.g.
`pip install langroid[litellm]` or `poetry add langroid[litellm]`

Then run this script as follows:

`python examples/basic/chat-local.py`

"""
import langroid as lr
import langroid.language_models as lm

# Assume you've run `ollama pull mistral` to spin up `mistral` locally.
# Notes:
# - we use `lm.OpenAIGPTConfig` to incidate this config is for LLMs served
#    at OpenAI-compatible endpoints)
# - if you omit `chat_model` below, it defaults to OpenAI GPT4-turbo,
#   or you can explicitly specify it as `lm.OpenAIChatModel.GPT4` or `lm.OpenAIChatModel.GPT4_TURBO`

llm_config = lm.OpenAIGPTConfig(
    chat_model="litellm/ollama/mistral",
    max_output_tokens=200,
    chat_context_length=2048,  # adjust based on your local LLM params
)

# Alternatively, if you've used ooba or other lib to spin up a Local LLM
# at an OpenAI-compatible endpoint, say http://localhost:8000, you can set the
# `chat_model` as follows (note you have to prefix it with 'local'):
# llm_config = lm.OpenAIGPTConfig(
#     chat_model="local/localhost:8000"
# )
# If the endpoint is listening at https://localhost:8000/v1, you must include the `v1`
# at the end, e.g. chat_model="local/localhost:8000/v1"

agent_config = lr.ChatAgentConfig(
    llm=llm_config,
    system_message="""Be helpful but very very concise""",
)

agent = lr.ChatAgent(agent_config)

task = lr.Task(agent)

task.run()
