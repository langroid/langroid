# Local embeddings provision via llama.cpp server

As of Langroid v0.30.0, you can use llama.cpp as provider of embeddings
to any of Langroid's vector stores, allowing access to a wide variety of
GGUF-compatible embedding models, e.g. [nomic-ai's Embed Text V1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF).

When defining a VecDB, you can provide an instance of 
`LlamaCppServerEmbeddingsConfig` to the VecDB config to instantiate
the llama.cpp embeddings server handler.

To configure the `LlamaCppServerEmbeddingsConfig`, there are several
parameters that should be adjusted, these are:

```python
embed_cfg = LlamaCppServerEmbeddingsConfig(
    api_base="your-address-here", # IP + Port, e.g. localhost:5001
    dims=768,  # Change this to match the dimensions of your embedding model
    context_length=2048, # Change to match the config of the model.
    batch_size=2048, # Safest to ensure this matches context_length
    )
```

The above configuration is sufficient for a server running the example
nomic embedding model with the command:
```
./llama-server -ngl 100 -c 2048 -m ~/nomic-embed-text-v1.5.Q8_0.gguf --host IP_ADDRESS --port PORT --embeddings -b 2048 -ub 2048
```

An example setup can be found inside [examples/docqa/chat.py](https://github.com/langroid/langroid/blob/main/examples/docqa/chat.py).

