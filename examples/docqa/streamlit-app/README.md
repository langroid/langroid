# Basic example: chat with a document using Langroid with local LLM or OpenAI LLM

Bare-bones example of an app that combines:
- Langroid `DocChatAgent` for RAG
- StreamLit for webapp/UI
to let you ask questions about the contents of a file (pdf, txt, docx, md, html).

## Instructions
Run this from the root of the `langroid-examples` repo. Assuming you already have a virtual env in 
which you have installed `langroid`, the only additional requirement is to run:

``` 
pip install streamlit
```
Then run the application like this:
```
streamlit run examples/docqa/streamlit-app/app.py
```
In the sidebar you can specify a local LLM, or leave it blank to use the OpenAI 
GPT4-Turbo model. 

## Local LLMs

Here are instructions to use this with a Local LLM spun up via [ollama](https://github.com/jmorganca/ollama)
(see their GitHub repo for more details but the below should suffice):

(1) Mac: Install latest ollama, then do this:
```bash
ollama pull mistral:7b-instruct-v0.2-q4_K_M
```

(2) Ensure you've installed the `litellm` extra with Langroid, e.g.
```bash
pip install langroid[litellm]
``` 
or if you use the `pyproject.toml` in this repo you can simply use `poetry install`

In the app sidebar you can then specify the model as:
```
litellm/ollama/mistral:7b-instruct-v0.2-q4_K_M
```

Other possibilities for local_model are:
- If instead of ollama (perhaps using ooba text-generation-webui)
  you've spun up your local LLM to listen at an OpenAI-Compatible Endpoint
  like `localhost:8000`, then you can use -m local/localhost:8000
- If the endpoint is listening at https://localhost:8000/v1, you must include the `v1`
- If the endpoint is http://127.0.0.1:8000, use -m local/127.0.0.1:8000

And so on. The above are few-shot examples for you. You get the idea.

## Limitations

- Streaming does not currently work
- Conversation is not accumulated
- Source, Extract evidence-citation is only displayed in terminal/console, to reduce clutter in the UI.

## Credits
Code adapted from Prashant Kumar's example in [`lancedb/vectordb-recipies`](https://github.com/lancedb/vectordb-recipes)