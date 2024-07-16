---
title: 'Chat formatting in Local LLMs'
draft: true
date: 2024-01-25
authors: 
  - pchalasani
categories:
  - langroid
  - prompts
  - llm
  - local-llm
comments: true
---


In an (LLM performance) investigation, details matter!

And assumptions kill (your LLM performance).

I'm talking about chat/prompt formatting, especially when working with Local LLMs.

TL/DR -- details like chat formatting matter a LOT,
and trusting that the local LLM API is doing it correctly may be a mistake,
leading to inferior results.

<!-- more -->

🤔Curious? Here are some notes from the trenches when we built an app
(https://github.com/langroid/langroid/blob/main/examples/docqa/chat-multi-extract-local.py)
based entirely on a locally running Mistral-7b-instruct-v0.2  
(yes ONLY 7B parameters, compared to 175B+ for GPT4!)
that leverages Langroid Multi-agents, Tools/Function-calling and RAG to
reliably extract structured information from a document,
where an Agent is given a spec of the desired structure, and it generates
questions for another Agent to answer using RAG.

🔵LLM API types: generate and chat
LLMs are typically served behind two types of APIs endpoints:
⏺ A "generation" API, which accepts a dialog formatted as a SINGLE string, and
⏺ a "chat" API, which accepts the dialog as a LIST,
and as convenience formats it into a single string before sending to the LLM.

🔵Proprietary vs Local LLMs
When you use a proprietary LLM API (such as OpenAI or Claude), for convenience
you can use their "chat" API, and you can trust that it will format the dialog
history correctly (or else they wouldn't be in business!).

But with a local LLM, you have two choices of where to send the dialog history:
⏺ you could send it to the "chat" API and trust that the server will format it correctly,
⏺ or you could format it yourself and send it to the "generation" API.

🔵Example of prompt formatting?
Suppose your system prompt and dialog look like this:

System Prompt/Instructions: when I give you a number, respond with its double
User (You): 3
Assistant (LLM): 6
User (You): 9

Mistral-instruct models expect this chat to be formatted like this
(note that the system message is combined with the first user message):
"<s>[INST] when I give you a number, respond with its double 3 [/INST] 6 [INST] 9 [/INST]"

🔵Why does it matter?
It matters A LOT -- because each type of LLM (llama2, mistral, etc) has
been trained and/or fine-tuned on chats formatted in a SPECIFIC way, and if you
deviate from that, you may get odd/inferior results.

🔵Using Mistral-7b-instruct-v0.2 via oobabooga/text-generation-webui
"Ooba" is a great library (https://github.com/oobabooga/text-generation-webui)
that lets you spin up an OpenAI-like API server for
local models, such as llama2, mistral, etc. When we used its chat endpoint
for a Langroid Agent, we were getting really strange results,
with the LLM sometimes thinking it is the user! 😧

Digging in, we found that their internal formatting template was
wrong, and it was formatting the system prompt as if it's
the first user message -- this leads to the LLM interpreting the first user
message as an assistant response, and so on -- no wonder there was role confusion!

💥Langroid solution:
To avoid these issues, in Langroid we now have a formatter
(https://github.com/langroid/langroid/blob/main/langroid/language_models/prompt_formatter/hf_formatter.py)
that retrieves the HuggingFace tokenizer for the LLM and uses
its "apply_chat_template" method to format chats.
This gives you control over the chat format and you can use the "generation"
endpoint of the LLM API instead of the "chat" endpoint.

Once we switched to this, results improved dramatically 🚀

Be sure to checkout Langroid https://github.com/langroid/langroid

#llm #ai #opensource 