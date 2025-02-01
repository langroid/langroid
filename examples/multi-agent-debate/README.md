Debate System Using LLM Agents
==============================

Overview
--------
This project is a debate system powered by LLMs using Langroid, enabling structured debates on various topics 
such as AI in healthcare, education, intellectual property, and societal biases. 
The program creates and manages agents that represent opposing sides of a debate, 
interact with users, and provide constructive feedback based on established debate criteria.

New Topics and Pro and Con Side System messages can be manually configured by updating or modifying the 
system_messages.json File. 
"pro_ai": {
        "topic": "Your New TOPIC",
        "message": " YOUR Prompt"
    },
"con_ai": {
        "topic": "Your New TOPIC",
        "message": " YOUR CON or opposing Prompt"
        }

Features
--------
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

File Structure
--------------
- main.py: The entry point of the application. Initializes the system, configures agents, and starts the debate loop.
- config.py: Provides functions for configuring global settings and LLM-specific parameters.
- model.py: Pydantic model for system_messages.json
- system_messages.json: Topic Titles and system_messages for pro and con agents. You can add more topics and their
respective pro and con system messages here. The system_messages has a statement: 
"Limit responses to MAXIMUM 2 points expressed as single sentences." Please change or delete it for a realistic debate. 
- system_message.py: Global system messages
- utils.py: User Prompts and other helper functions
- generation_config_models.py: pydantic model for generation_config.json
- generation_config.json: LLM generation parameters
- main_chainlit.py: The entry point of the Chainlit version of the application
- chainlit_utils.py: Relevant chainlit utility functions.
The system dynamically updates user selection with the topics from this file. 

Getting Started
---------------
Prerequisites
1. Python 3.8+
2. Langroid Framework: Install Langroid with necessary dependencies:
   pip install "langroid[litellm]"
3. Setup the following env variables in the .env File in the root of your repo
or set them on your terminal.
       export OPENAI_API_KEY=OPEN AI KEY
       export GEMINI_API_KEY=GEMiNi API KEY
       export METAPHOR_API_KEY=METAPHOR_API_KEY
4. Please read the following page for more information:
   https://langroid.github.io/langroid/quick-start/setup/

Usage
-----
Run the CLI Application
Start the application from the root of the langroid repo with:
   python examples/multi-agent-debate/main.py

Options
- Debug Mode: Run the program with debug logs for detailed output.
  python examples/multi-agent-debate/main.py --debug
- Disable Caching: Avoid using cached responses for LLM interactions.
  python examples/multi-agent-debate/main.py --nocache

Run the Chainlit App
chainlit run examples/multi-agent-debate/main_chainlit.py


Interaction
1. Decide if you want to you use same LLM for all agents or different ones
2. Decide if you want autonomous debate between AI Agents or user vs. AI Agent. 
3. Select a debate topic.
4. Choose your side (Pro or Con).
5. Engage in a debate by providing arguments and receiving responses from agents.
6. Request feedback at any time by typing `f`.
7. Decide if you want the Metaphor Search to run to find Topic relevant web links
   and summarize them. 
8. Decide if you want to chat with the documents extracted from URLs found to learn more about the Topic.
9. End the debate manually by typing "done". If you decide to chat with the documents, you can end session
by typing `x`

Feedback Criteria
-----------------
The feedback mechanism evaluates debates based on:
1. Clash of Values
2. Argumentation
3. Cross-Examination
4. Rebuttals
5. Persuasion
6. Technical Execution
7. Adherence to Debate Etiquette
8. Final Focus

License
-------
This project is licensed under the MIT License.
