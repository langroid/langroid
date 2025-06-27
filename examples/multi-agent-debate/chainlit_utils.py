import logging
from typing import Optional, Tuple

import chainlit as cl
from config import MODEL_MAP
from models import SystemMessages
from utils import extract_topics

DEFAULT_TURN_COUNT = 2
DEFAULT_TIMEOUT = 100

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_boolean_response(response: str) -> bool:
    """
    Convert a user response into a boolean value.
    Args:
        response (str): User input as "yes" or "no".
    Returns:
        bool: True for "yes", False for "no".
    """
    if response == "yes":
        return True
    elif response == "no":
        return False
    raise ValueError("Invalid response: expected 'yes' or 'no'.")


async def handle_boolean_response(res, default=False):
    """
    Handle the user's response from an AskActionMessage.

    Args:
        res (dict): The response dictionary from AskActionMessage.
        default (bool): The default value to return in case of errors or timeouts.

    Returns:
        bool: Parsed boolean response from the user.
    """
    if res:
        try:
            user_choice = res.get("payload", {}).get("value", "").lower()
            return parse_boolean_response(user_choice)
        except ValueError:
            await cl.Message(
                content=f"Unexpected response. Defaulting to '{default}'."
            ).send()
            return default
    # Default if no response or timeout
    await cl.Message(
        content=f"You didn't respond in time. Defaulting to '{default}'."
    ).send()
    return default


async def is_same_llm_for_all_agents() -> bool:
    """
    Ask the user if they want to use the same LLM for all agents.

    Returns:
        bool: True if yes, False if no. Timeout or no response is defaulted to False.
    """

    # Create a Chainlit action message with a timeout
    ask_message = cl.AskActionMessage(
        content=f"Do you want to use the same LLM for all agents?\n\n(If you do not respond within {DEFAULT_TIMEOUT} "
        f"seconds, we will default to selecting individual LLMs.)",
        actions=[
            cl.Action(name="yes", payload={"value": "yes"}, label="Yes"),
            cl.Action(name="no", payload={"value": "no"}, label="No"),
        ],
        timeout=DEFAULT_TIMEOUT,
    )

    res = await ask_message.send()

    # Override the timeout before Chainlit sends its message
    if not res:
        await ask_message.remove()  # Removes the pending action before timeout triggers
        res = {"payload": {"value": "no"}}  # Auto-select "No"

    user_selection = await handle_boolean_response(res, default=False)

    await cl.Message(
        content=(
            "You have chosen to proceed with the same LLM for all agents."
            if user_selection
            else "You have chosen to select individual LLMs for each agent."
        )
    ).send()

    return user_selection


async def select_max_debate_turns() -> int:
    """
    Ask the user to select the maximum number of turns for debates.
    Returns:
        int: The number of debate turns.
    """
    ask_message = cl.AskActionMessage(
        content=f"How many turns should the debates take?\n\n(If you do not respond within {DEFAULT_TIMEOUT} "
        f"seconds, we will default to selecting 2 turns.)",
        actions=[
            cl.Action(name="2", payload={"value": "2"}, label="2"),
            cl.Action(name="4", payload={"value": "4"}, label="4"),
            cl.Action(name="8", payload={"value": "8"}, label="8"),
            cl.Action(name="16", payload={"value": "16"}, label="16"),
        ],
        timeout=DEFAULT_TIMEOUT,
    )

    res = await ask_message.send()

    # Prevents Chainlit's default timeout message
    if not res:
        await ask_message.remove()
        res = {"payload": {"value": "2"}}  # Default to 2 turns

    try:
        turns = int(res["payload"]["value"])
        await cl.Message(content=f"You selected {turns} turns for the debate.").send()
        return turns
    except (ValueError, KeyError):
        await cl.Message(content="Invalid input. Defaulting to 2 turns.").send()
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
    # Create the AskActionMessage and send it
    ask_message = cl.AskActionMessage(
        content=f"Should the Pro and Con agents debate autonomously?\n\n(If you do not respond within {DEFAULT_TIMEOUT} "
        f"seconds, we will default to autonomous debate.)",
        actions=[
            cl.Action(name="yes", payload={"value": "yes"}, label="Yes"),
            cl.Action(name="no", payload={"value": "no"}, label="No"),
        ],
        timeout=DEFAULT_TIMEOUT,
    )

    res = await ask_message.send()

    # # Prevents Chainlit's default timeout message
    if not res:
        await ask_message.remove()
        res = {"payload": {"value": "no"}}  # Auto-select "No"

    user_selection = await handle_boolean_response(res, default=False)

    await cl.Message(
        content=(
            "You have chosen to proceed with autonomous debate"
            if user_selection
            else "You have chosen to engage in debate with an AI agent"
        )
    ).send()

    print("The user selected to proceed with the debate")
    return user_selection


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
    LLM_DELEGATE_FLAG, system_messages: "SystemMessages"
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
    if LLM_DELEGATE_FLAG:
        side = "pro"
    else:
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
    response = await cl.AskUserMessage(
        content=prompt_text, timeout=DEFAULT_TIMEOUT
    ).send()
    if response:
        try:
            user_input = response["output"].strip()
            topic_index = int(user_input) - 1
            if 0 <= topic_index < len(topics):
                selected_topic = topics[topic_index]
                logger.info(f"Selected topic: {selected_topic[0]}")
                await cl.Message(
                    content=f"You have chosen the following debate topic: {selected_topic[0]}"
                ).send()
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
        selected_topic = topics[0]
        await cl.Message(
            content=f"You didn't respond in time. The system has chosen the following default Topic:  {selected_topic[0]}"
        ).send()
        return selected_topic


async def is_metaphor_search_key_set() -> bool:
    """
    Prompt the user for confirmation about Metaphor Search API keys.

    Returns:
        bool: True if the user confirms they have an API key, otherwise False.
    """
    ask_message = cl.AskActionMessage(
        content=f"Do you have an API Key for Metaphor Search?,\n\n(If you do not respond within {DEFAULT_TIMEOUT} "
        f"seconds, we will default to selecting that you don't have the API Key or dont' want to search)",
        actions=[
            cl.Action(name="yes", payload={"value": "yes"}, label="Yes"),
            cl.Action(name="no", payload={"value": "no"}, label="No"),
        ],
        timeout=DEFAULT_TIMEOUT,
    )

    res = await ask_message.send()

    # Prevents Chainlit's default timeout message
    if not res:
        await ask_message.remove()
        res = {"payload": {"value": "no"}}  # Auto-select "No"

    user_selection = await handle_boolean_response(res, default=False)

    await cl.Message(
        content=(
            "You have chosen to use the Metaphor Search for Research Agent."
            if user_selection
            else "You have chosen that Metaphor Search API key is not available."
        )
    ).send()

    return user_selection


async def is_url_ask_question(topic_name: str) -> bool:
    """
    Prompt the user for confirmation if they want to Q/A by loading the URL documents into vecdb.

    Args:
        topic_name (str): The topic name for the question.

    Returns:
        bool: True if the user confirms for Q/A, otherwise False.
    """
    ask_message = cl.AskActionMessage(
        content=f"Would you like to chat with web searched documents for more information on {topic_name},"
        f"\n\n(If you do not respond within {DEFAULT_TIMEOUT} "
        f"seconds, we will default to selecting that you don't want to chat with the documents)",
        actions=[
            cl.Action(name="yes", payload={"value": "yes"}, label="Yes"),
            cl.Action(name="no", payload={"value": "no"}, label="No"),
        ],
        timeout=DEFAULT_TIMEOUT,
    )

    res = await ask_message.send()

    # Prevents Chainlit's default timeout message
    if not res:
        await ask_message.remove()
        res = {"payload": {"value": "no"}}  # Auto-select "No"

    user_selection = await handle_boolean_response(res, default=False)

    await cl.Message(
        content=(
            f"You have chosen to chat with web-searched documents using RAG for {topic_name}."
            if user_selection
            else f"You have chosen NOT to chat with web-searched documents for {topic_name}."
        )
    ).send()

    return user_selection
