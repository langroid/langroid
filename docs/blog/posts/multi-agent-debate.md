---
title: 'Multi Agent Debate and Education Platform'
draft: false
date: 2025-01-03
authors: 
  - adamshams
categories:
  - langroid
  - llm
  - local-llm
  - chat
comments: true
---
Introducing an AI-powered multi-agent debate platform leveraging cutting-edge Large Language Models (LLMs) and 
the powerful Langroid multi-agent LLM framework. This platform simulates structured debates between users 
and multiple AI agents on critical AI ethics topics. In addition, the platform can easily adapt to any topic by simply adding pro and con system message making it a powerful eduction
tool to learn about diverse topics. 

The system employs a multi-agent design using Langroid powerful framework, where each agent embodies a specific ethical
perspective, creating realistic and dynamic interactions. 
Agents are prompt-engineered and role-tuned to align with their assigned ethical stance, 
ensuring thoughtful and structured debates. The system is adaptable to user needs and they can select:
- Issues such as AI’s impact on healthcare, intellectual property, societal biases, and its role as an educator, 
choosing to argue either the Pro or Con side.
- Engage in manual debate with a pro or con agent or watch it autonomously.
- Adjust # of turns
- Easily add Topics and system prompts
- Experiment with responses from multiple models, including OpenAI, Google, and Mistral, 
to explore diverse viewpoints.

The Platform utilizes four different types of agents:
1. Pro Agent: Dynamically created based on the selected Topics' Pro system message
2. Con Agent: Dynamically created based on the selected Topics' Pro system message
3. Feedback agent evaluates arguments in real-time and provides final assessments based on 
clarity, strength, and relevance. Generates the summary of Pro and Con arguments and declares a winner of the debate.
4. Research Agent: Conducts neural searches using the Metaphor Search API and the MetaphorSearchTool from Langroid. 
It dynamically adopts to user selected topics' pro and con system messages for performing neural searches, 
identify web references, and finally produce a summary of relevant arguments for and against the topic

### Features
- Multiple Debate Topics:
  - AI in Healthcare
  - AI and Intellectual Property
  - AI and Societal Biases
  - AI as an Educator
- Agent-Based Interaction:
  - Pro and Con agents for each topic simulate structured debate arguments.
- Configurable to use different LLMs from OPENAI, Google, & Mistral: 
  -       1: gpt-4o
          2: gpt-4
          3: gpt-4o-mini
          4: gpt-4-turbo
          5: gpt-4-32k
          6: gpt-3.5-turbo-1106 
          7: Mistral: mistral:7b-instruct-v0.2-q8_0a
          8: Gemini:gemini-2.0-flash
          9: Gemini:gemini-1.5-flash
          10: Gemini:gemini-1.5-flash-8b
          11: Gemini:gemini-1.5-pro
- Feedback Mechanism:
  - Provides structured feedback on debate performance based on key criteria.
- Interactive or Autonomous Mode:
  - Users can either control interactions manually or let agents autonomously continue debates.
- Research Agent conducting neural search using the topic pro and con system messages to find 
Web references and then generate pro and con arguments. Metaphor Search is utilized to find references
on the web. MetaphorSearchAPI is required for this part. The system can skip if the user doesn't have the API Key

```python
METAPHOR_SEARCH_AGENT_SYSTEM_MESSAGE_TEMPLATE = """
            There are 2 STEPs. Your Goal is to execute both of them. 
            STEP 1:  Run MetaphorSearchTool

            Use the TOOL {metaphor_tool_name} to search the web for 5 references for Pro: {pro_message}
            and Con: {con_message}.     
            YOUR GOAL IS TO FIND GOOD REFERENCES FOR BOTH SIDES OF A DEBATE. 
            Be very CONCISE in your responses, use 5-7 sentences. 
            show me the SOURCE(s) and EXTRACT(s) and summary
            in this format:

            <your answer here>
            Here are additional references using Metaphor Search to improve your knowledge of the subject:

            M1: SOURCE: https://journalofethics.ama-assn.org/article/should-artificial-intelligence-augment-
            medical-decision-making-case-autonomy-algorithm/2018-09
            EXTRACT: Discusses the ethical implications of AI in medical decision-making and 
            the concept of an autonomy algorithm.
            SUMMARY: This article explores the ethical considerations of integrating AI into medical decision-making 
            processes, emphasizing the need for autonomy and ethical oversight.

            M2: SOURCE: ...
            EXTRACT: ...
            SUMMARY:

            DO NOT MAKE UP YOUR OWN SOURCES; ONLY USE SOURCES YOU FIND FROM A WEB SEARCH. 
            ENSURE STEP 1 IS COMPLETED BEFORE STARTING STEP 2

            STEP 2: Argue Pro and Con Cases
            As an expert debater, your goal is to eloquently argue for both the Pro and Con cases
            using the references from web-search SOURCES generated in Step 1 and properly cite the Sources in BRACKETS 
            (e.g., [SOURCE])
            Write at least 5 sentences for each side.

            ENSURE BOTH STEP 1 and 2 are completed. 
            After all STEPs are completed, use the `{done_tool_name}` tool to end the session   
            """

```
The METAPHOR_SEARCH_AGENT_SYSTEM_MESSAGE_TEMPLATE message is dynamically updated based on the pro and con 
system messages configured in system_messages.json

```python
from langroid.agent.tools.metaphor_search_tool import MetaphorSearchTool
from langroid.agent.tools.orchestration import DoneTool

def generate_metaphor_search_agent_system_message(system_messages, pro_key, con_key):
    return METAPHOR_SEARCH_AGENT_SYSTEM_MESSAGE_TEMPLATE.format(
        metaphor_tool_name=MetaphorSearchTool.name(),
        pro_message=system_messages.messages[pro_key].message,
        con_message=system_messages.messages[con_key].message,
        done_tool_name=DoneTool.name()
  )
```
### File Structure
1. `main.py`: The entry point of the application. Initializes the system, configures agents, and starts the debate loop.
2. `config.py`: Provides functions for configuring global settings and LLM-specific parameters.Additional models
can be easily configured in config.py and adjusting the prompt by modifying the `select_model`  in `util.py`
```python
MODEL_MAP = {
    "1": lm.OpenAIChatModel.GPT4o,
    "2": lm.OpenAIChatModel.GPT4,
    "3": lm.OpenAIChatModel.GPT4o_MINI,
    "4": lm.OpenAIChatModel.GPT4_TURBO,
    "5": lm.OpenAIChatModel.GPT4_32K,
    "6": lm.OpenAIChatModel.GPT3_5_TURBO,
    "7": "ollama/mistral:7b-instruct-v0.2-q8_0",
    "8": "gemini/" + lm.GeminiModel.GEMINI_2_FLASH,
    "9": "gemini/" + lm.GeminiModel.GEMINI_1_5_FLASH,
    "10": "gemini/" + lm.GeminiModel.GEMINI_1_5_FLASH_8B,
    "11": "gemini/" + lm.GeminiModel.GEMINI_1_5_PRO,
}
```
3. `model.py`: Pydantic model for `system_messages.json`
4. `system_messages.json`: Topic Titles and system messages for pro and con agents. You can add more topics and their 
respective pro and con system messages here. The system messages have a statement: "Limit responses to MAXIMUM 2 points 
expressed as single sentences." Please change or delete it for a realistic debate. New Topics can be easily added by modifying a system_messages.json. Pro and Con Side System messages can be configured 
without any code changes. 

```python
"pro_id": {
        "topic": "Your New TOPIC",
        "message": " System Prompt to argue for the topic"
    },
"con_id": {
        "topic": "Your New TOPIC",
        "message": "System prompt to argue against the topic"
        }
```

5. `system_message.py`: Global system messages
6. `utils.py`: User Prompts and other helper functions
7. `generation_config_models.py`: Pydantic model for `generation_config.json`
```python
class GenerationConfig(BaseModel):
    """Represents configuration for text generation."""
    max_output_tokens: int = Field(default=10_000, ge=1, description="Maximum output tokens.")
    min_output_tokens: int = Field(default=1, ge=0, description="Minimum output tokens.")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Sampling temperature.")
    seed: Optional[int] = Field(default=42,
        description="Seed for reproducibility. If set, ensures deterministic outputs for the same input."
    )
```
8. `generation_config.json`: LLM generation parameters can be modified here:
```
{
  "max_output_tokens": 10000,
  "min_output_tokens": 1,
  "temperature": 0.7,
  "seed": 42
}
```
The system dynamically updates user selection with the topics from this file. 

### Getting Started
Prerequisites
1. Python 3.8+
2. Langroid Framework: Install Langroid with necessary dependencies:
```
pip install "langroid[litellm]"
```
3. Set up the following env variables in the .env File in the root of your repo
or set them on your terminal.
```
       export OPENAI_API_KEY=OPENAI_API_KEY
       export GEMINI_API_KEY=GEMiNi_API_KEY
       export METAPHOR_API_KEY=METAPHOR_API_KEY
```
4. Please read the following page [Langroid setup](https://langroid.github.io/langroid/quick-start/setup/)

### Usage
Run the CLI Application
Start the application from the root of the langroid repo with:
```
   python examples/multi-agent-debate/main.py
```
#### Options
- Debug Mode: Run the program with debug logs for detailed output.
  python examples/multi-agent-debate/main.py --debug
- Disable Caching: Avoid using cached responses for LLM interactions.
  python examples/multi-agent-debate/main.py --nocache

### Interaction
1. Decide if you want to you use same LLM for all agents or different ones
2. Decide if you want autonomous debate between AI Agents or user vs. AI Agent. 
3. Select a debate topic.
4. Choose your side (Pro or Con).
5. Engage in a debate by providing arguments and receiving responses from agents.
6. Request feedback at any time by typing `f`.
7. Decide if you want the Metaphor Search to run to find Topic relevant web links
   and summarize them. 
8. End the debate manually by typing `done`.

### Why Langroid was chosen?

I chose Langroid framework because it's the most principled multi-agent programming framework 
inspired by Actor framework.
Prior to using Langroid, I developed a multi-agent debate system, but I had to manage states of communication between
debating agents and manage all the interactions with LLMs. Langroid allowed me seamlessly integrate multiple LLMs,
easily create agents, tasks, and task delegations. Here is a code example of how simple it was to create and delegate
and run tasks and access agent's user_messages:

```python
    # Set up langroid tasks and run the debate
    user_task = Task(user_agent, interactive=interactive_setting, restart=False)
    ai_task = Task(ai_agent, interactive=False, single_round=True)
    user_task.add_sub_task(ai_task)
    if not llm_delegate:
        user_task.run(user_agent.user_message, turns=max_turns)
    else:
        user_task.run("get started", turns=max_turns)
    
```
Tasks can be easily chained as sub-tasks of an orchestrating agent. In this case user_task could be Pro or Con depending 
on the user selection. 

if you want to use custom tools/functions with LLMs or use Langroid 
provided its only a line of code using `enable_message`. Here is an example using MetaphorSearchTool and DoneTool:
```python
        metaphor_search_agent.enable_message(MetaphorSearchTool)
        metaphor_search_agent.enable_message(DoneTool)
```

Overall I had a delightful time, and it was a great learning experience using Langroid and recommend using it for any 
projects that need to utilize LLMs. I am already working on a few Langroid `DocChat` based 
information retrieval and research systems for use in medical applications, and I'm hoping to contribute more soon. 


