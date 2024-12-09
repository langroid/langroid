import typer
from rich.prompt import Prompt, Confirm
import json
from config import get_base_llm_config, get_global_settings
from models import SystemMessages, Message
from typing import List, Tuple, Optional, Literal, Any
import logging
import langroid as lr
import langroid.utils.logging
import langroid.agent.tools
import langroid.agent.tools.google_search_tool
from langroid.language_models import OpenAIGPTConfig
from langroid import ChatAgent, ChatAgentConfig
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import DoneTool

# Initialize typer application
app = typer.Typer()

# Set up a logger for this module
# Configure logging with a StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Set a logging format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the handler to the logger
if not logger.handlers:  # Avoid adding multiple handlers
    logger.addHandler(console_handler)


def load_system_messages(file_path: str) -> SystemMessages:
    """Load and validate system messages from a JSON file.

    Reads the JSON file containing system messages, maps each entry to a
    `Message` object, and wraps the result in a `SystemMessages` object.

    Args:
        file_path (str): The path to the JSON file containing system messages.

    Returns:
        SystemMessages: A `SystemMessages` object containing validated messages.

    Raises:
        IOError: If the file cannot be read or found.
        json.JSONDecodeError: If the JSON file is not properly formatted.
        Exception: For any other unexpected errors during processing.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data: Any = json.load(f)
        # Map dictionaries to Message objects
        messages = {key: Message(**value) for key, value in data.items()}
        return SystemMessages(messages=messages)
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise IOError(f"Could not find the file: {file_path}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file: {file_path}")
        raise json.JSONDecodeError(
            f"Invalid JSON format in file: {file_path}", e.doc, e.pos
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error loading system messages: {e}")
        raise


def extract_topics(system_messages: SystemMessages) -> List[Tuple[str, str, str]]:
    """Extract unique debate topics from the SystemMessages object.

    Processes the `SystemMessages` object to identify debate topics by pairing
    `pro_` and `con_` keys. Ensures each topic is represented only once.

    Args:
        system_messages (SystemMessages): The object containing system messages
            with `pro_` and `con_` topic keys.

    Returns:
        List[Tuple[str, str, str]]: A list of tuples, where each tuple contains:
            - topic_name (str): The name of the debate topic.
            - pro_key (str): The key for the pro side of the debate.
            - con_key (str): The key for the con side of the debate.
    """
    topics: List[Tuple[str, str, str]] = []
    for key, message in system_messages.messages.items():
        if key.startswith("pro_"):  # Process only "pro_" keys to avoid duplicates
            con_key = key.replace("pro_", "con_", 1)  # Match "con_" dynamically
            if con_key in system_messages.messages:  # Ensure "con_" exists
                topics.append((message.topic, key, con_key))
    return topics


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
        f"Which side would you like to debate on?\n1. Pro-{topic_name}\n2. Con-{topic_name}",
        choices=["1", "2"],
        default="1",
    )
    return "pro" if side == "1" else "con"


def is_llm_delegate() -> bool:
    """Prompt the user to decide on LLM delegation.

    Asks the user whether the LLM should autonomously continue the debate
    without requiring user input.

    Returns:
        bool: True if the user chooses LLM delegation, False otherwise.
    """
    return Confirm.ask(
        "Would you like the LLM to autonomously continue the debate without "
        "waiting for user input? or ask for your input first time?",
        default=False,
    )


def run_debate() -> None:
    """Execute the main debate logic.

    Orchestrates the debate process, including setup, user input, LLM agent
    interactions, and final feedback. Handles both user-guided and LLM-
    delegated debates.

    Raises:
        Exception: If an unexpected error occurs during the debate process.
    """
    try:
        # Get global settings
        global_settings = get_global_settings(nocache=True)
        langroid.utils.configuration.set_global(global_settings)

        # Get base LLM configuration
        agent_config: AgentConfig = get_base_llm_config()
        system_messages: SystemMessages = load_system_messages(
            "system_messages.json"
        )
        llm_delegate: bool = is_llm_delegate()

        # Select topic and sides
        selected_topic_tuple: Optional[tuple] = select_debate_topic(system_messages)
        if not selected_topic_tuple:
            logger.error("No topic selected. Exiting.")
            return
        topic_name, pro_key, con_key = selected_topic_tuple
        side: str = select_side(topic_name)

        # Prompt for the number of debate turns
        max_turns: int = int(
            Prompt.ask(
                "How many turns should the debate continue for?",
                default="4",
            )
        )
        # Create agents for pro, con, and feedback
        pro_agent = lr.ChatAgent(
            lr.ChatAgentConfig(
                llm=agent_config,
                system_message=system_messages.messages[pro_key].message
            )
        )

        con_agent = lr.ChatAgent(
            lr.ChatAgentConfig(
                llm=agent_config,
                system_message=system_messages.messages[con_key].message
            )
        )

        feedback_agent = lr.ChatAgent(
            lr.ChatAgentConfig(
                llm=agent_config,
                system_message=system_messages.messages["feedback"].message
            )
        )

        logger.info("Pro, Con, and feedback agents Created.")

        # Determine user's side
        if side == "pro":
            user_agent, ai_agent = pro_agent, con_agent
            user_side, ai_side = "Pro", "Con"
        else:
            user_agent, ai_agent = con_agent, pro_agent
            user_side, ai_side = "Con", "Pro"

        logger.info(
            f"Starting debate on topic: {topic_name}, taking the {user_side} side. "
            f"LLM Delegate: {llm_delegate}"
        )

        is_user_turn: bool = True

        if side == "pro":
            pro_agent.clear_history()
            logger.info(f"\n{side} Agent ({topic_name}):\n")
            if llm_delegate:
                logger.info("Autonomous Debate Selected")
            else:
                user_input = Prompt.ask(
                    "Your argument (or type 'f' for feedback, 'done' to end):"
                )
                pro_agent.llm_response(user_input)
            pro_agent.enable_message(DoneTool)
            pro_agent_task = lr.Task(pro_agent, interactive=False)
            con_agent_task = lr.Task(con_agent, interactive=False, single_round=True)
            pro_agent_task.add_sub_task(con_agent_task)
            pro_agent_task.run(turns=max_turns)
        else:
            con_agent.clear_history()
            logger.info(f"\n{side} Agent ({topic_name}):\n")
            if llm_delegate:
                logger.info("Autonomous Debate Selected")
            else:
                user_input = Prompt.ask(
                    "Your argument (or type 'f' for feedback, 'done' to end):"
                )
                pro_agent.llm_response(user_input)
            con_agent.enable_message(DoneTool)
            con_agent_task = lr.Task(pro_agent, interactive=False)
            pro_agent_task = lr.Task(con_agent, interactive=False, single_round=True)
            con_agent_task.add_sub_task(pro_agent_task)
            pro_agent_task.run(turns=max_turns)

        feedback_agent.llm_response(
            f"Summarize the debate and declare a winner.\n{con_agent.message_history}"
        )

    except Exception as e:
        logger.error(f"Unexpected error during debate: {e}")
        raise


@app.command()
def main():
    run_debate()


if __name__ == "__main__":
    app()
