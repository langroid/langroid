import typer
from rich.prompt import Prompt, Confirm
import json
from agents import create_agent
from config import get_base_llm_config, get_global_settings
from models import SystemMessages, Message
from typing import List, Tuple, Optional, Literal, Any
import logging
import langroid.utils.logging

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
        "waiting for user input?",
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

        # Create agents for pro, con, and feedback
        pro_agent: Agent = create_agent(
            agent_config, system_messages.messages[pro_key].message
        )
        con_agent: Agent = create_agent(
            agent_config, system_messages.messages[con_key].message
        )
        feedback_agent: Agent = create_agent(
            agent_config, system_messages.messages["feedback"].message
        )
        logger.info("Pro, Con, and feedback agents started.")

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

        # Prompt for the number of debate turns
        max_turns: int = int(
            Prompt.ask(
                "How many turns should the debate continue for?",
                default="4",
            )
        )
        student_arguments: List[str] = []
        ai_arguments: List[str] = []
        is_user_turn: bool = True

        for turn in range(max_turns):
            if llm_delegate:
                # Autonomous LLM debate
                current_agent = user_agent if is_user_turn else ai_agent
                agent_role = user_side if is_user_turn else ai_side
                opponent_arguments = (
                    ai_arguments if is_user_turn else student_arguments
                )
                context = (
                    "\n".join(opponent_arguments[-1:])
                    if opponent_arguments
                    else "Start of debate."
                )

                # Prepare the full prompt
                full_context = (
                    f"{current_agent.system_message}\n\n"
                    f"Please ensure the response includes rebuttals to all "
                    f"opponent's arguments:\n{context}"
                )

                logger.info(f"\n{agent_role} Agent ({topic_name}):\n")
                print(f"\n{agent_role} Agent ({topic_name}):\n", end="", flush=True)
                response = current_agent.llm_response(full_context)
                if response and response.content:
                    argument = response.content.strip()
                    if is_user_turn:
                        student_arguments.append(argument)
                    else:
                        ai_arguments.append(argument)
                else:
                    print(f"\n{agent_role} Agent did not respond.")
                is_user_turn = not is_user_turn
            else:
                if is_user_turn:
                    # User's turn
                    user_input = Prompt.ask(
                        "Your argument (or type 'f' for feedback, 'done' to end):"
                    )
                    if user_input.lower() == "f":
                        # Provide feedback during the debate
                        feedback_content = "\n".join(
                            student_arguments + ai_arguments
                        )
                        print("\nFeedback:\n", end="", flush=True)
                        final_feedback = feedback_agent.llm_response(
                            f"Provide feedback on the debate so far.\n{feedback_content}"
                        )
                        print()  # Newline after feedback
                    elif user_input.lower() == "done":
                        logger.info("Debate ended by user.")
                        break
                    else:
                        student_arguments.append(user_input)
                        is_user_turn = not is_user_turn  # Switch to AI's turn
                else:
                    # AI Agent's turn
                    context = (
                        student_arguments[-1]
                        if student_arguments
                        else "Start of debate."
                    )
                    full_context = (
                        f"{ai_agent.system_message}\n\n"
                        f"Opponent's argument:\n{context}"
                    )
                    print(f"\n{ai_side} Agent ({topic_name}):\n", end="", flush=True)
                    response = ai_agent.llm_response(full_context)
                    if response and response.content:
                        argument = response.content.strip()
                        ai_arguments.append(argument)
                    else:
                        print(f"\n{ai_side} Agent did not respond.")
                    is_user_turn = not is_user_turn

        # Final feedback
        final_feedback_content = "\n".join(student_arguments + ai_arguments)
        print("\nFinal Feedback:\n", end="", flush=True)
        final_feedback = feedback_agent.llm_response(
            f"Summarize the debate and declare a winner.\n{final_feedback_content}"
        )

    except Exception as e:
        logger.error(f"Unexpected error during debate: {e}")
        raise


@app.command()
def main():
    run_debate()


if __name__ == "__main__":
    app()
