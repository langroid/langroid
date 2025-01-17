import chainlit as cl
import logging
from models import SystemMessages
from typing import Tuple, Optional
from config import MODEL_MAP
from utils import extract_topics

DEFAULT_TURN_COUNT = 2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_boolean_response(response: str) -> bool:
    """
    Parse user's response into a boolean value.
    Args:
        response (str): User input as a string.
    Returns:
        bool: True for yes/true, False for no/false.
    """
    response = response.strip().lower()
    if response in ["yes", "y", "true"]:
        return True
    elif response in ["no", "n", "false"]:
        return False
    raise ValueError("Invalid boolean response")


async def is_same_llm_for_all_agents() -> bool:
    """
    Ask the user if they want to use the same LLM for all agents.
    Returns:
        bool: True if yes, False if no.
    """
    same_llm = await cl.AskUserMessage(
        content="Do you want to use the same LLM for all agents? (yes/no)", timeout=10
    ).send()
    if same_llm:
        try:
            result = parse_boolean_response(same_llm["output"])
            if result:
                await cl.Message(
                    content="Great! Same LLM will be used for all agents."
                ).send()
            else:
                await cl.Message(
                    content="Ok. Different LLMs will be configured for the agents."
                ).send()
            return result
        except ValueError:
            await cl.Message(
                content="Invalid response. Please answer yes or no."
            ).send()
            return await is_same_llm_for_all_agents()  # Retry on invalid input
    else:
        await cl.Message(
            content="You didn't respond in time. Defaulting to 'no'."
        ).send()
        return False


async def select_max_debate_turns() -> int:
    """
    Ask the user to select the maximum number of turns for debates.
    Returns:
        int: The number of debate turns.
    """
    max_turns = await cl.AskUserMessage(
        content="How many turns should the debates take?", timeout=10
    ).send()
    if max_turns:
        try:
            turns = int(max_turns["output"].strip())
            await cl.Message(
                content=f"You selected {turns} turns for the debate."
            ).send()
            return turns
        except ValueError:
            await cl.Message(
                content="Invalid input. Please provide a valid number. Defaulting to 2 turns."
            ).send()
            return DEFAULT_TURN_COUNT
    else:
        await cl.Message(
            content="You didn't respond in time. Defaulting to 2 turns."
        ).send()
        return DEFAULT_TURN_COUNT


async def select_model(config_agent_name: str) -> str:
    """
    Prompts the user to select an LLM model for the specified agent.
    Args:
        config_agent_name (str): The name of the agent being configured.
    Returns:
        str: The selected model key from MODEL_MAP.
    """
    # Model selections for user
    llm_options = {
        "1": "GPT-4o",
        "2": "GPT-4",
        "3": "GPT-4o-MINI",
        "4": "GPT-4-TURBO",
        "5": "GPT-4-32K",
        "6": "GPT-3.5-TURBO",
        "7": "Mistral 7b-instruct",
        "8": "Gemini 2.0 Flash",
        "9": "Gemini 1.5 Flash",
        "10": "Gemini 1.5 Flash 8B",
        "11": "Gemini 1.5 Pro",
    }

    # Prepare the user prompt
    options_text = "\n".join([f"{key}: {value}" for key, value in llm_options.items()])
    prompt_text = f"Select a Model for {config_agent_name}:\n{options_text}\nEnter your choice (1-{len(llm_options)}):"

    # Prompt the user for model selection
    response = await cl.AskUserMessage(content=prompt_text, timeout=20).send()
    if response:
        try:
            selected_option = response["output"].strip()
            if selected_option in MODEL_MAP:
                selected_model = MODEL_MAP[selected_option]
                await cl.Message(
                    content=f"You selected: {llm_options[selected_option]}"
                ).send()
                return selected_option
            else:
                await cl.Message(
                    content="Invalid selection. Please enter a valid number."
                ).send()
                return await select_model(config_agent_name)  # Retry on invalid input
        except Exception as e:
            await cl.Message(content=f"An error occurred: {e}").send()
            return await select_model(config_agent_name)  # Retry on error
    else:
        await cl.Message(
            content="You didn't respond in time. Defaulting to GPT-4o."
        ).send()
        return "1"  # Default to GPT-4o


async def is_llm_delegate() -> bool:
    """
    Ask the user if the Pro and Con agents should debate autonomously.
    Returns:
        bool: True if yes, False if no.
    """
    llm_delegate = await cl.AskUserMessage(
        content="Should the Pro and Con agents debate autonomously? (yes/no)",
        timeout=10,
    ).send()
    if llm_delegate:
        llm_delegate = llm_delegate["output"].strip().lower()
        is_yes = llm_delegate in ["yes", "y", "true"]
        is_no = llm_delegate in ["no", "n", "false"]

        if is_yes:
            await cl.Message(
                content="Great! The agents will debate autonomously."
            ).send()
            return True
        elif is_no:
            await cl.Message(
                content="Okay, agents will not debate autonomously."
            ).send()
            return False
        else:
            await cl.Message(
                content="I didn't understand your response. Please answer yes or no."
            ).send()
            return await is_llm_delegate()  # Retry on invalid input
    else:
        await cl.Message(
            content="You didn't respond in time. Defaulting to 'no'."
        ).send()
        return False


async def select_side(topic_name: str) -> str:
    """
    Prompt the user to select a pro or con side in the debate
    Args:
        topic_name (str): The name of the debate topic.
    Returns:
        str: The selected debate side, either "pro" or "con".
    """
    response = await cl.AskUserMessage(
        content=f"Which side would you like to debate on?\n1. Pro-{topic_name}\n2. Con-{topic_name}",
        timeout=20,
    ).send()

    if response:
        side_choice = response["output"].strip()
        if side_choice in ["1", "2"]:
            return "pro" if side_choice == "1" else "con"
        else:
            await cl.Message(
                content="Invalid selection. Please choose 1 for Pro or 2 for Con."
            ).send()
            return await select_side(topic_name)  # Retry on invalid input
    else:
        await cl.Message(
            content="You didn't respond in time. Defaulting to 'pro'."
        ).send()
        return "pro"  # Default to "pro" if no response


async def select_topic_and_setup_side(
    system_messages: "SystemMessages",
) -> Tuple[str, str, str, str]:
    """
    Prompt the user to select a debate topic and sets up the respective side.
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
    selected_topic_tuple = await select_debate_topic(
        system_messages
    )  # Assuming this is an async function
    if not selected_topic_tuple:
        logger.error("No topic selected. Exiting.")
        raise ValueError("No topic selected.")

    topic_name, pro_key, con_key = selected_topic_tuple
    side = await select_side(topic_name)
    return topic_name, pro_key, con_key, side


async def select_debate_topic(system_messages: "SystemMessages") -> Optional[tuple]:
    """
    Prompt the user to select a debate topic dynamically loaded from  SystemMessages.
    Args:
        system_messages (SystemMessages): The object containing debate topics.
    Returns:
        Optional[tuple]: A tuple containing:
            - topic_name (str): The selected topic's name.
            - pro_key (str): The key for the pro side of the debate.
            - con_key (str): The key for the con side of the debate.
            Returns None if no topics are available or an error occurs.
    """
    # Extract topics from SystemMessages
    topics = extract_topics(system_messages)
    if not topics:
        logger.error("No topics found in the SystemMessages object.")
        await cl.Message(content="No debate topics are available.").send()
        return None

    # Prepare the topic choices for user selection
    topic_choices = "\n".join(
        [f"{i + 1}. {topic[0]}" for i, topic in enumerate(topics)]
    )
    prompt_text = (
        f"Select a debate topic:\n{topic_choices}\nEnter your choice (1-{len(topics)}):"
    )

    # Prompt the user for topic selection
    response = await cl.AskUserMessage(content=prompt_text, timeout=20).send()
    if response:
        try:
            user_input = response["output"].strip()
            topic_index = int(user_input) - 1
            if 0 <= topic_index < len(topics):
                selected_topic = topics[topic_index]
                logger.info(f"Selected topic: {selected_topic[0]}")
                return selected_topic
            else:
                await cl.Message(
                    content="Invalid selection. Please choose a valid topic number."
                ).send()
                return await select_debate_topic(
                    system_messages
                )  # Retry on invalid input
        except ValueError:
            await cl.Message(
                content="Invalid input. Please enter a number corresponding to a topic."
            ).send()
            return await select_debate_topic(system_messages)  # Retry on invalid input
    else:
        await cl.Message(
            content="You didn't respond in time. No topic selected."
        ).send()
        return None


async def is_metaphor_search_key_set() -> bool:
    """
    Prompt the user for confirmation about Metaphor Search API keys.
    Returns:
        bool: True if the user confirms they have an API key, otherwise False.
    """
    response = await cl.AskUserMessage(
        content="Do you have an API Key for Metaphor Search? (yes/no)",
        timeout=20,
    ).send()

    if response:
        try:
            result = parse_boolean_response(response["output"])
            if result:
                await cl.Message(
                    content="Great! You confirmed having an API Key for Metaphor Search."
                ).send()
            else:
                await cl.Message(
                    content="You confirmed not having an API Key for Metaphor Search."
                ).send()
            return result
        except ValueError:
            await cl.Message(
                content="Invalid response. Please answer with yes or no."
            ).send()
            return await is_metaphor_search_key_set()  # Retry on invalid input
    else:
        await cl.Message(
            content="You didn't respond in time. Defaulting to 'no'."
        ).send()
        return False


async def is_url_ask_question(topic_name: str) -> bool:
    """
    Prompt the user for confirmation if they want to Q/A by loading  the URLS docs into vecdb
    Returns:
        bool: True if the user confirms for Q/A, otherwise False.
    """
    response = await cl.AskUserMessage(
        content=f"Would you like to Chat with documents found through Search for more information on the {topic_name}",
        timeout=20,
    ).send()

    if response:
        try:
            result = parse_boolean_response(response["output"])
            if result:
                await cl.Message(
                    content="Great! You confirmed to Chat with documents"
                ).send()
            else:
                await cl.Message(
                    content="You confirmed not to chat with documents."
                ).send()
            return result
        except ValueError:
            await cl.Message(
                content="Invalid response. Please answer with yes or no."
            ).send()
            return await is_metaphor_search_key_set()  # Retry on invalid input
    else:
        await cl.Message(
            content="You didn't respond in time. Defaulting to 'no'."
        ).send()
        return False
