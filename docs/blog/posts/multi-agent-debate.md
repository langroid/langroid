---
title: 'Multi Agent Debate and Education Platform'
draft: false
date: 2025-02-04
authors: 
  - adamshams
categories:
  - langroid
  - llm
  - local-llm
  - chat
comments: true
---

## Introduction
Have you ever imagined a world where we can debate complex issues with Generative AI agents taking a distinct 
stance and backing their arguments with evidence? Some will change your mind, and some will reveal the societal biases 
on which each distinctive Large Language Model (LLM) is trained on. Introducing an [AI-powered debate platform](https://github.com/langroid/langroid/tree/main/examples/multi-agent-debate) that brings 
this imagination to reality, leveraging diverse LLMs and the Langroid multi-agent programming framework. The system enables users to engage in structured debates with an AI taking the opposite stance (or even two AIs debating each other), using a multi-agent architecture with Langroid's powerful framework, where each agent embodies a specific ethical perspective, creating realistic and dynamic interactions. 
Agents are prompt-engineered and role-tuned to align with their assigned ethical stance, 
ensuring thoughtful and structured debates. 

<!-- more -->

My motivations for creating this platform included: 

  - A debate coach for underserved students without access to traditional resources. 
  - Tool for research and generating arguments from authentic sources. 
  - Create an adaptable education platform to learn two sides of the coin for any topic.
  - Reduce echo chambers perpetuated by online algorithms by fostering two-sided debates on any topic, promoting education and awareness around misinformation. 
  - Provide a research tool to study the varieties of biases in LLMs that are often trained on text reflecting societal biases. 
  - Identify a good multi-agent framework designed for programming with LLMs.


## Platform Features:
### Dynamic Agent Generation:
The platform features five types of agents: Pro, Con, Feedback, Research, and Retrieval Augmented Generation (RAG) Q&A. 
Each agent is dynamically generated using role-tuned and engineered prompts, ensuring diverse and engaging interactions.
#### Pro and Con Agents: 
These agents engage in the core debate, arguing for and against the chosen topic. 
Their prompts are carefully engineered to ensure they stay true to their assigned ethical stance.
#### Feedback Agent: 
This agent provides real-time feedback on the arguments and declares a winner. The evaluation criteria are based on the well-known Lincoln–Douglas debate format, and include:

  - Clash of Values 
  - Argumentation 
  - Cross-Examination 
  - Rebuttals 
  - Persuasion 
  - Technical Execution 
  - Adherence to Debate Etiquette 
  - Final Focus
#### Research Agent: 
This agent has the following functionalities:

  - Utilizes the `MetaphorSearchTool` and the `Metaphor` (now called `Exa`) Search API to conduct web searches combined with
Retrieval Augmented Generation (RAG) to relevant web references for user education about the selected topic. 
  - Produces a summary of arguments for and against the topic.
  - RAG-based document chat with the resources identified through Web Search. 
#### RAG Q&A Agent:

  - Provides Q&A capability using a RAG based chat interaction with the resources identified through Web Search.
The agent utilizes `DocChatAgent` that is part of Langroid framework which orchestrates all LLM interactions. 
  - Rich chunking parameters allows the user to get optimized relevance results. Check out `config.py`for details.

### Topic Adaptability:
Easily adaptable to any subject by simply adding pro and con system messages. This makes it a versatile tool for
exploring diverse topics and fostering critical thinking. Default topics cover ethics and use of AI for the following:
  - Healthcare
  - Intellectual property 
  - Societal biases 
  - Education
### Autonomous or Interactive:
Engage in manual debate with a pro or con agent or watch it autonomously while adjusting number of turns.

### Diverse LLM Selection Adaptable per Agent: 
Configurable to select from diverse commercial and open source models: OpenAI, Google, and Mistral 
to experiment with responses for diverse perspectives. Users can select a unique LLM for each agent. 
       
### LLM Tool/Function Integration: 
Utilizes LLM tools/functions features to conduct semantic search using Metaphor Search API and summarizes the pro and 
con perspectives for education.

### Configurable LLM Parameters: 
Parameters like temperature, minimum and maximum output tokens, allowing for customization of the AI's responses.
Configurable LLM parameters like temperature, min & max output tokens. For Q&A with the searched resources, several
parameters can be tuned in the `config` to enhance response relevance.

### Modular Design: 
Reusable code and modularized for other LLM applications.


## Interaction
1. Decide if you want to you use same LLM for all agents or different ones
2. Decide if you want autonomous debate between AI Agents or user vs. AI Agent. 
3. Select a debate topic.
4. Choose your side (Pro or Con).
5. Engage in a debate by providing arguments and receiving responses from agents.
6. Request feedback at any time by typing `f`.
7. Decide if you want the Metaphor Search to run to find Topic relevant web links
   and summarize them. 
8. Decide if you want to chat with the documents extracted from URLs found to learn more about the Topic.
9. End the debate manually by typing `done`. If you decide to chat with the documents, you can end session
by typing `x`

## Why was Langroid chosen?
I chose Langroid framework because it's a principled multi-agent programming framework inspired by the Actor framework.
Prior to using Langroid, I developed a multi-agent debate system, however, I had to write a lot of tedious code to manage states of communication between
debating agents, and the user interactions with LLMs. Langroid allowed me to seamlessly integrate multiple LLMs,
easily create agents, tasks, and attach sub-tasks. 

### Agent Creation Code Example

```python
   def create_chat_agent(name: str, llm_config: OpenAIGPTConfig, system_message: str) -> ChatAgent:
   
    return ChatAgent(
        ChatAgentConfig(
            llm=llm_config,
            name=name,
            system_message=system_message,
        )
    )
```
#### Sample Pro Topic Agent Creation

```python
 
    pro_agent = create_chat_agent(
        "Pro",
        pro_agent_config,
        system_messages.messages[pro_key].message + DEFAULT_SYSTEM_MESSAGE_ADDITION,
    )
    
```
The `Task` mechanism in Langroid provides a robust mechanism for managing complex interactions within multi-agent 
systems. `Task` serves as a container for managing the flow of interactions between different agents
(such as chat agents) and attached sub-tasks.`Task` also helps with turn-taking, handling responses, 
and ensuring smooth transitions between dialogue states. Each Task object is responsible for coordinating responses 
from its assigned agent, deciding the sequence of responder methods (llm_response, user_response, agent_response), 
and managing transitions between different stages of a conversation or debate. Each agent can focus on its specific 
role while the task structure handles the overall process's orchestration and flow, allowing a clear separation of 
concerns. The architecture and code transparency of Langroid's framework make it an incredible candidate for 
applications like debates where multiple agents must interact dynamically and responsively
based on a mixture of user inputs and automated responses.

### Task creation and Orchestration Example

```python
    user_task = Task(user_agent, interactive=interactive_setting, restart=False)
    ai_task = Task(ai_agent, interactive=False, single_round=True)
    user_task.add_sub_task(ai_task)
    if not llm_delegate:
        user_task.run(user_agent.user_message, turns=max_turns)
    else:
        user_task.run("get started", turns=max_turns)
    
```
Tasks can be easily set up as sub-tasks of an orchestrating agent. In this case user_task could be Pro or Con depending 
on the user selection. 

If you want to build custom tools/functions or use Langroid provided it is only a line of code using
`agent.enable_messaage`. Here is an example of `MetaphorSearchTool` and `DoneTool`. 
```python
        metaphor_search_agent.enable_message(MetaphorSearchTool)
        metaphor_search_agent.enable_message(DoneTool)
```

Overall I had a great learning experience using Langroid and recommend using it for any projects 
that need to utilize LLMs. I am already working on a few Langroid based information retrieval and research systems 
for use in medicine and hoping to contribute more soon. 

### Bio

I'm a high school senior at Khan Lab School located in Mountain View, CA where I host a student-run Podcast known as the
Khan-Cast. I also enjoy tinkering with interdisciplinary STEM projects. You can reach me on [LinkedIn](https://www.linkedin.com/in/adamshams/).

