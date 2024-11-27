"""
tasks.py

Defines tasks for agents in the debate system. Each task is associated
with a specific agent and role. Tasks allow agents to interact within
the debate structure.
"""

import logging
import langroid as lr

# Set up a logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_task(agent: lr.ChatAgent, name: str) -> lr.Task:
    """Creates a generic task for a given agent.

    Args:
        agent (ChatAgent): The agent assigned to this task.
        name (str): Name of the task.

    Returns:
        Task: A configured task for the agent.
    """
    logger.info("Creating task: %s", name)

    return lr.Task(
        agent,
        name=name,
        single_round=True,
        interactive=False,
    )


# Tasks for AI in Healthcare
def create_pro_ai_task(agent: lr.ChatAgent) -> lr.Task:
    """Creates a Pro-AI task for healthcare debates."""
    logger.info("Creating Pro-AI Task for Healthcare.")
    return create_task(agent, "Pro_AI_Agent_Healthcare")


def create_con_ai_task(agent: lr.ChatAgent) -> lr.Task:
    """Creates a Con-AI task for healthcare debates."""
    logger.info("Creating Con-AI Task for Healthcare.")
    return create_task(agent, "Con_AI_Agent_Healthcare")


# Tasks for AI and Intellectual Property Rights
def create_pro_ip_task(agent: lr.ChatAgent) -> lr.Task:
    """Creates a Pro-IP task for intellectual property debates."""
    logger.info("Creating Pro-IP Task.")
    return create_task(agent, "Pro_IP_Agent")


def create_con_ip_task(agent: lr.ChatAgent) -> lr.Task:
    """Creates a Con-IP task for intellectual property debates."""
    logger.info("Creating Con-IP Task.")
    return create_task(agent, "Con_IP_Agent")


# Tasks for AI and Societal Biases
def create_pro_bias_task(agent: lr.ChatAgent) -> lr.Task:
    """Creates a Pro-Bias task for societal bias debates."""
    logger.info("Creating Pro-Bias Task.")
    return create_task(agent, "Pro_Bias_Agent")


def create_con_bias_task(agent: lr.ChatAgent) -> lr.Task:
    """Creates a Con-Bias task for societal bias debates."""
    logger.info("Creating Con-Bias Task.")
    return create_task(agent, "Con_Bias_Agent")


# Tasks for AI as an Educator
def create_pro_edu_task(agent: lr.ChatAgent) -> lr.Task:
    """Creates a Pro-Education task for debates on AI in education."""
    logger.info("Creating Pro-Education Task.")
    return create_task(agent, "Pro_Edu_Agent")


def create_con_edu_task(agent: lr.ChatAgent) -> lr.Task:
    """Creates a Con-Education task for debates on AI in education."""
    logger.info("Creating Con-Education Task.")
    return create_task(agent, "Con_Edu_Agent")


# Feedback Task
def create_feedback_task(agent: lr.ChatAgent) -> lr.Task:
    """Creates a Feedback task for structured debates."""
    logger.info("Creating Feedback Task.")
    return create_task(agent, "Feedback_Agent")
