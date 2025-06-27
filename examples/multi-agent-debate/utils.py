import logging
import re
from typing import List, Literal, Optional, Tuple

from models import SystemMessages
from rich.prompt import Confirm, Prompt

from langroid.utils.logging import setup_logger

DEFAULT_TURN_COUNT = 2

# set info logger
logger = setup_logger(__name__, level=logging.INFO, terminal=True)


def extract_topics(system_messages: SystemMessages) -> List[Tuple[str, str, str]]:
    """Extract unique debate topics from the SystemMessages object.

    Processes the `SystemMessages` object to identify debate topics by pairing
    `pro_` and `con_` keys. Ensures each topic is represented only once.

    Args:
        system_messages (SystemMessages): The object containing system messages
            with `pro_` and `con_` topic keys.

    Returns:
        List[Tuple[str, str, str]]: A list of tuples,
        where each tuple contains:
            - topic_name (str): The name of the debate topic.
            - pro_key (str): The key for the pro side of the debate.
            - con_key (str): The key for the con side of the debate.
    """
    topics: List[Tuple[str, str, str]] = []
    for key, message in system_messages.messages.items():
        # Process only "pro_" keys to avoid duplicates
        if key.startswith("pro_"):
            con_key = key.replace("pro_", "con_", 1)  # Match "con_" dynamically
            if con_key in system_messages.messages:  # Ensure "con_" exists
                topics.append((message.topic, key, con_key))
    return topics


def select_model(config_agent_name: str) -> str:
    """
    Prompt the user to select an OpenAI or Gemini model
    for the specified agent.

    This function prompts the user to select an option from
    a list of available models.
    The user's input corresponds to a predefined choice, which is
    then returned as a string representing the selected option.

    Args:
        config_agent_name (str): The name of the agent being configured,
        used in the prompt to personalize the message.

    Returns:
        str: The user's selected option as a string, corresponding to one of the
             predefined model choices (e.g., "1", "2", ..., "10").
    """
    return Prompt.ask(
        f"Select a Model for {config_agent_name}:\n"
        "1: gpt-4o\n"
        "2: gpt-4\n"
        "3: gpt-4o-mini\n"
        "4: gpt-4-turbo\n"
        "5: gpt-4-32k\n"
        "6: gpt-3.5-turbo-1106\n"
        "7: Mistral: mistral:7b-instruct-v0.2-q8_0a\n"
        "8: Gemini: gemini-2.0-flash\n"
        "8: Gemini: gemini-1.5-flash\n"
        "9: Gemini: gemini-1.5-flash-8b\n"
        "10: Gemini: gemini-1.5-pro\n",
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        default="1",
    )


def select_debate_topic(system_messages: SystemMessages) -> Optional[tuple]:
    """Prompt the user to select a debate topic from SystemMessages.

    Dynamically loads debate topics from the SystemMessages object, displays
    the options to the user, and prompts them to select a topic.

    Args:
        system_messages (SystemMessages): The object containing debate topics.

    Returns:
        Optional[tuple]: A tuple containing:
            - topic_name (str): The selected topic's name.
            - pro_key (str): The key for the pro side of the debate.
            - con_key (str): The key for the con side of the debate.
            Returns None if no topics are available or an error occurs.
    """
    topics = extract_topics(system_messages)

    if not topics:
        logger.error("No topics found in the JSON file.")
        return None

    # Prepare topic choices for user selection
    topic_choices = "\n".join(
        [f"{i + 1}. {topic[0]}" for i, topic in enumerate(topics)]
    )
    user_input = Prompt.ask(
        f"Select a debate topic:\n{topic_choices}",
        choices=[str(i + 1) for i in range(len(topics))],
        default="1",
    )
    topic_index = int(user_input) - 1

    selected_topic = topics[topic_index]
    logger.info(f"Selected topic: {selected_topic[0]}")
    return selected_topic


def select_side(topic_name: str) -> Literal["pro", "con"]:
    """Prompt the user to select their side in the debate.

    Presents the user with a choice to debate on either the pro or con side
    of the given topic.

    Args:
        topic_name (str): The name of the debate topic.

    Returns:
        Literal["pro", "con"]: The selected debate side.
    """
    side = Prompt.ask(
        f"Which side would you like to debate on?\n1. Pro-{topic_name}\n2. "
        f"Con-{topic_name}",
        choices=["1", "2"],
        default="1",
    )
    return "pro" if side == "1" else "con"


def select_topic_and_setup_side(
    system_messages: SystemMessages,
) -> Tuple[str, str, str, str]:
    """Prompt the user to select a debate topic and sets up the respective side.

    This function handles the user interaction for selecting a debate topic and the side
    (Pro or Con) they want to argue. It validates that a topic is selected and raises an
    exception if the topic is not available.

    Args:
        system_messages (SystemMessages): The object containing system messages with respective
                                          debate topics.

    Returns:
        Tuple[str, str, str, str]: A tuple containing:
            - topic_name (str): The name of the selected debate topic.
            - pro_key (str): The key for the Pro side of the selected topic.
            - con_key (str): The key for the Con side of the selected topic.
            - side (str): The user's selected side, either "pro" or "con".

    Raises:
        ValueError: If no topic is selected or no topics are available in the provided
                    `system_messages`.
    """
    selected_topic_tuple = select_debate_topic(system_messages)
    if not selected_topic_tuple:
        logger.error("No topic selected. Exiting.")
        raise ValueError("No topic selected.")

    topic_name, pro_key, con_key = selected_topic_tuple
    side = select_side(topic_name)
    return topic_name, pro_key, con_key, side


def is_llm_delegate() -> bool:
    """Prompt the user to decide on LLM delegation.

    Asks the user whether the LLM should autonomously continue the debate
    without requiring user input.

    Returns:
        bool: True if the user chooses LLM delegation, otherwise return False.
    """
    return Confirm.ask(
        "Should the Pro and Con agents debate autonomously?",
        default=False,
    )


def is_metaphor_search_key_set() -> bool:
    """Prompt the user confirmation about metaphorSearch API keys.

    Asks the user to confirm the metaphorSearch API keys.

    Returns:
        bool: True if the user chooses LLM delegation, otherwise return False.
    """
    return Confirm.ask(
        "Do you have an API Key for Metaphor Search?",
        default=False,
    )


def is_same_llm_for_all_agents() -> bool:
    """Prompt the user to decide if same LLM should be used for all agents.

    Asks the user whether the same LLM should be configured for all agents.

    Returns:
        bool: True if the user chooses same LLM for all agents, otherwise return False.
    """
    # Ask the user if they want to use the same LLM configuration for all agents
    return Confirm.ask(
        "Do you want to use the same LLM for all agents?",
        default=True,
    )


def select_max_debate_turns() -> int:
    # Prompt for number of debate turns
    while True:
        max_turns = Prompt.ask(
            "How many turns should the debate continue for?",
            default=str(DEFAULT_TURN_COUNT),
        )
        try:
            return int(max_turns)
        except ValueError:
            return DEFAULT_TURN_COUNT


def extract_urls(message_history):
    """
    Extracts all URLs from the given message history content and returns them in the format [url1, url2, ..., urln].

    Parameters:
        message_history (list): A list of LLMMessage objects containing message history.

    Returns:
        str: A string representation of a list of URLs.
    """
    # Extract content only from non-system messages
    content = " ".join(
        message.content
        for message in message_history
        if hasattr(message, "content") and message.content and message.role != "system"
    )

    # Extract URLs from content
    urls = re.findall(r"https?://\S+", content)
    return urls  # Return the list of URLs directly


def is_url_ask_question(topic_name: str) -> bool:
    """Prompt the user to decide to ask questions from the searched URL docs.

    Asks the user whether they want to ask questions from the searched URL docs?

    Returns:
        bool: True if the user chooses to ask questions from searched url docs., otherwise return False.
    """
    return Confirm.ask(
        f"Would you like to Chat with documents found through Search for more information on the {topic_name}",
        default=False,
    )
