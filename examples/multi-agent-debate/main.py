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
from langroid.agent.tools.google_search_tool import GoogleSearchTool

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
if not logger.handlers:
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
        bool: True if the user chooses LLM delegation, otherwise return False.
    """
    return Confirm.ask(
        "Would you like the LLM to autonomously continue the debate without "
        "waiting for user input? or ask for your input first time?",
        default=False,
    )
def is_google_api_key_configured() -> bool:
    """Prompt the user to validate if the Google API Key and CSE ID are configured.

    Asks the user whether they have configured the Google API key and CSE ID.

    Returns:
        bool: True if the user has configured the required env variables for Google API Key and CSE ID, otherwise
        return False.
    """
    return Confirm.ask(
        "Do you have a Google API key and configured? For example, GOOGLE_API_KEY=your-key and GOOGLE_CSE_ID=your-cse-id",
        default=False,
    )


def run_debate() -> None:
    """Execute the main debate logic.

    Orchestrates the debate process, including setup, user input, LLM agent
    interactions, and final feedback. Handles both user-guided and LLM-
    delegated debates.


    This function:
    1. Loads global settings and the base LLM configurations.
    2. Prompts the user to select a debate topic and a side(Pro or Con).
    3. Sets up pro, con, and feedback agents.
    4. Runs the debate for a specified number of turns, either interactively
       or autonomously.
    5. Provides a feedback summary at the end.
    6. Optionally validates references if a Google API key is configured. Creates a references validation agent.
    Utilizes the langroid GoogleSearchTool.
    """
    global_settings = get_global_settings(nocache=True)
    langroid.utils.configuration.set_global(global_settings)

    # Get base LLM configuration

    pro_agent_config: OpenAIGPTConfig = get_base_llm_config("for Pro Agent")
    con_agent_config: OpenAIGPTConfig = get_base_llm_config("for Con Agent")
    agent_config: OpenAIGPTConfig = get_base_llm_config("feedback and googleSearch")
    system_messages: SystemMessages = load_system_messages("system_messages.json")
    llm_delegate: bool = is_llm_delegate()

    # Select topic and sides
    selected_topic_tuple = select_debate_topic(system_messages)
    if not selected_topic_tuple:
        logger.error("No topic selected. Exiting.")
        return
    topic_name, pro_key, con_key = selected_topic_tuple
    side: str = select_side(topic_name)

    # Prompt for number of debate turns
    max_turns: int = int(
        Prompt.ask("How many turns should the debate continue for?", default="4")
    )

    # Create agents for pro, con, and feedback
    pro_agent = ChatAgent(
        ChatAgentConfig(
            llm=pro_agent_config,
            name="Pro",
            system_message=system_messages.messages[pro_key].message,
        )
    )

    con_agent = ChatAgent(
        ChatAgentConfig(
            llm=con_agent_config,
            name="Con",
            system_message=system_messages.messages[con_key].message,
        )
    )

    feedback_agent = ChatAgent(
        ChatAgentConfig(
            llm=agent_config,
            name="feedback",
            system_message=f"""
            You are an expert and experienced judge specializing in Lincoln-Douglas style debates. 
            Your goal is to evaluate the debate thoroughly based on the following criteria:
            1. Clash of Values: Assess how well each side upholds their stated value (e.g., justice, morality) 
               and how effectively they compare and prioritize values.
            2. Argumentation: Evaluate the clarity, organization, and logical soundness of each side's case structure, 
               contentions, and supporting evidence.
            3. Cross-Examination: Judge the effectiveness of questioning and answering during cross-examination.
            4. Rebuttals: Analyze how well each side refutes their opponent's arguments.
            5. Persuasion: Assess communication quality, tone, rhetorical effectiveness, and emotional/ethical appeals.
            6. Technical Execution: Identify if major arguments were addressed or dropped and check consistency.
            7. Debate Etiquette: Evaluate professionalism, respect, and demeanor.
            8. Final Focus: Judge the strength of closing speeches, how well they summarize the case, 
            and justify a winner.
            Provide constructive feedback for each debater, 
            summarizing their performance and declaring a winner with justification.
            """

        ),
        )
    )

    logger.info("Pro, Con, and feedback agents created.")

    # Determine user's side and assign user_agent and ai_agent based on user selection
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

    # Prepare user agent history by clearing the memory
    user_agent.clear_history()
    logger.info(f"\n{user_side} Agent ({topic_name}):\n")

    # Determine if the debate is autonomous or the user input for one side
    if llm_delegate:
        logger.info("Autonomous Debate Selected")
        interactive_setting = False
    else:
        interactive_setting = True
        user_input: str = Prompt.ask(
            "Your argument (or type 'f' for feedback, 'done' to end):"
        )
        user_agent.llm = None  # User message without LLM completion
        user_agent.llm_response(user_input)

    # Set up langroid tasks and run the debate
    user_agent.enable_message(DoneTool)
    user_task = Task(user_agent, interactive=interactive_setting, restart=False)
    ai_task = Task(ai_agent, interactive=False, single_round=True)
    user_task.add_sub_task(ai_task)
    user_task.run("get started", turns=max_turns)

    # Determine the last agent
    last_agent = user_agent if max_turns % 2 == 0 else ai_agent

    # Generate feedback summary and declare a winner using feedback agent
    validation_message: str = last_agent.message_history
    feedback_agent.llm_response(
        f"Summarize the debate and declare a winner.\n{validation_message}"
    )

    # If Google API is configured, run validation checks
    if is_google_api_key_configured():
        google_validation_agent = ChatAgent(
            ChatAgentConfig(
                llm=agent_config,
                name="validation",
                system_message=f"""
                        You are a helpful assistant. Extract all the links in references and then use the 
                        GoogleSearchTool to validate the references.
                        Please show a list of all provided references and then a list of validated ones.
                        DO NOT MAKE UP YOUR OWN SOURCES; ONLY USE SOURCES YOU FIND FROM A WEB SEARCH.
                        """
            )
        )
        google_validation_agent.clear_history()
        google_validation_agent.enable_message(GoogleSearchTool)
        google_validate_task = Task(google_validation_agent, interactive=False)
        google_validate_task.run(validation_message)


@app.command()
def main():
    run_debate()


if __name__ == "__main__":
    app()
