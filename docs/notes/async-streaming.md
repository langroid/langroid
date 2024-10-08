# Suppressing output in async, streaming mode

Available since version 0.18.0

When using an LLM API in streaming + async mode, you may want to suppress output,
especially when concurrently running multiple instances of the API.
To suppress output in async + stream mode, 
you can set the `async_stream_quiet` flag in [`LLMConfig`][langroid.language_models.base.LLMConfig]
to `True` (this is the default). 
Note that [`OpenAIGPTConfig`][langroid.language_models.openai_gpt.OpenAIGPTConfig]
inherits from `LLMConfig`, so you can use this flag with `OpenAIGPTConfig` as well:

```python
import langroid.language_models as lm
llm_config = lm.OpenAIGPTConfig(
    async_stream_quiet=True,
    ...
)
```

