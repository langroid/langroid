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


## Limitations

- Streaming does not currently work
- Conversation is not accumulated
- Source, Extract evidence-citation is only displayed in terminal/console, to reduce clutter in the UI.

## Credits
Code adapted from Prashant Kumar's example in [`lancedb/vectordb-recipies`](https://github.com/lancedb/vectordb-recipes)