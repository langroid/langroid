# Support for Open LLMs hosted on glhf.chat

Available since v0.23.0.

If you're looking to use Langroid with one of the recent performant Open LLMs,
such as `Qwen2.5-Coder-32B-Instruct`, you can do so using our glhf.chat integration.

See [glhf.chat](https://glhf.chat/chat/create) for a list of available models.

To run with one of these models, 
set the chat_model in the `OpenAIGPTConfig` to `"glhf/<model_name>"`, 
where model_name is hf: followed by the HuggingFace repo path, 
e.g. `Qwen/Qwen2.5-Coder-32B-Instruct`, 
so the full chat_model would be `"glhf/hf:Qwen/Qwen2.5-Coder-32B-Instruct"`.

Also many of the example scripts in the main repo (under the `examples` directory) can
be run with this and other LLMs using the model-switch cli arg `-m <model>`, e.g.

```bash
python3 examples/basic/chat.py -m glhf/hf:Qwen/Qwen2.5-Coder-32B-Instruct
```

Additionally, you can run many of the tests in the `tests` directory with this model
instead of the default OpenAI `GPT4o` using `--m <model>`, e.g. 

```bash
pytest tests/main/test_chat_agent.py --m glhf/hf:Qwen/Qwen2.5-Coder-32B-Instruct
```

For more info on running langroid with Open LLMs via other providers/hosting services,
see our
[guide to using Langroid with local/open LLMs](https://langroid.github.io/langroid/tutorials/local-llm-setup/#local-llms-hosted-on-glhfchat).

