Debate System Using LLM Agents
==============================

Overview
--------
This project is a debate system powered by LLMs using Langroid, enabling structured debates on various topics such as AI in healthcare, education, intellectual property, and societal biases. The program creates and manages agents that represent opposing sides of a debate, interact with users, and provide constructive feedback based on established debate criteria.


Features
--------
- Multiple Debate Topics:
  - AI in Healthcare
  - AI and Intellectual Property
  - AI and Societal Biases
  - AI as an Educator
- Agent-Based Interaction:
  - Pro and Con agents for each topic simulate structured debate arguments.
- Customizable LLM Configuration:
  - Support for different LLMs like GPT-4, GPT-3.5, and Mistral.
- Feedback Mechanism:
  - Provides structured feedback on debate performance based on key criteria.
- Interactive or Autonomous Mode:
  - Users can either control interactions manually or let agents autonomously continue debates.

File Structure
--------------
- main.py: The entry point of the application. Initializes the system, configures agents, and starts the debate loop.
- agents.py: Defines functions to create agents for each debate topic, representing the Pro and Con sides.
- tasks.py: Contains task definitions that encapsulate agent behavior during debates.
- config.py: Provides functions for configuring global settings and LLM-specific parameters.

Getting Started
---------------
Prerequisites
1. Python 3.8+
2. Langroid Framework: Install Langroid with all dependencies:
   pip install "langroid[all]"

Usage
-----
Run the Application
Start the application with:
   python main.py

Options
- Debug Mode: Run the program with debug logs for detailed output.
  python main.py --debug
- Disable Caching: Avoid using cached responses for LLM interactions.
  python main.py --nocache

Interaction
1. Select a debate topic.
2. Choose your side (Pro or Con).
3. Engage in a debate by providing arguments and receiving responses from agents.
4. Request feedback at any time by typing `f`.
5. End the debate manually by typing `done`.

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
"""
