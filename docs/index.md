# Langroid: Harness LLMs with Multi-Agent Programming

## The LLM Opportunity

Given the remarkable abilities of recent Large Language Models (LLMs), there
is an unprecedented opportunity to build intelligent applications powered by
this transformative technology. The top question for any enterprise is: how
best to harness the power of LLMs for complex applications? For technical and
practical reasons, building LLM-powered applications is not as simple as
throwing a task at an LLM-system and expecting it to do it.

## Langroid's Multi-Agent Programming Framework

We are convinced that effectively leveraging LLMs at scale requires a
*principled programming framework*. In particular, there is often a need to 
maintain multiple LLM conversations, each instructed in different 
ways, and "responsible" for different aspects of a task.

An *agent* is a convenient abstraction that encapsulates LLM conversation 
state, along with access to long-term memory (vector-stores) and tools (a.k.a functions 
or plugins). Thus a **Multi-Agent Programming** framework is a natural fit 
for complex LLM-based applications.

Langroid is the first Python LLM-application framework that was explicitly 
designed  with Agents as first-class citizens, and Multi-Agent Programming 
as the core  design principle. The framework is inspired by ideas from the 
[Actor Framework](https://en.wikipedia.org/wiki/Actor_model).

Langroid allows an intuitive definition of agents, tasks and task-delegation 
among agents. There is a principled mechanism to orchestrate multi-agent 
collaboration. Agents act as message-transformers, and take turns responding to (and
transforming) the current message. The architecture is lightweight, transparent, 
flexible, and allows other types of orchestration to be implemented.

We believe Langroid’s Multi-Agent Programming framework will become essential
for developers to productively build complex applications with LLMs. 

