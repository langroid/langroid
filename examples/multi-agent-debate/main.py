import logging
from typing import Any, List

import typer
from config import get_base_llm_config, get_global_settings, get_questions_agent_config
from models import SystemMessages, load_system_messages
from rich.prompt import Prompt
from system_messages import (
    DEFAULT_SYSTEM_MESSAGE_ADDITION,
    FEEDBACK_AGENT_SYSTEM_MESSAGE,
    generate_metaphor_search_agent_system_message,
)

# Import from utils.py
from utils import (
    extract_urls,
    is_llm_delegate,
    is_metaphor_search_key_set,
    is_same_llm_for_all_agents,
    is_url_ask_question,
    select_max_debate_turns,
    select_model,
    select_topic_and_setup_side,
)

import langroid as lr
from langroid import ChatDocument, Entity
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tools.metaphor_search_tool import MetaphorSearchTool
from langroid.agent.tools.orchestration import DoneTool
from langroid.language_models import OpenAIGPTConfig
from langroid.utils.logging import setup_logger


class MetaphorSearchChatAgent(ChatAgent):
    def handle_message_fallback(self, msg: str | ChatDocument) -> str | None:
        """Handle scenario where LLM did not generate any Tool"""
        if isinstance(msg, ChatDocument) and msg.metadata.sender == Entity.LLM:
            return f"""
            Have you presented pro and con arguments based on 
            your search results? If so, use the TOOL `{DoneTool.name()}` to indicate you're finished. 
            Otherwise, argue both sides and then send the `{DoneTool.name()}`
            """
        return None


# Initialize typer application
app = typer.Typer()

# set info logger
logger = setup_logger(__name__, level=logging.INFO, terminal=True)
logger.info("Starting multi-agent-debate")


def parse_and_format_message_history(message_history: List[Any]) -> str:
    """
    Parses and formats message history to exclude system messages
    and map roles to Pro/Con.

    Args:
        message_history (List[Any]): The full message history
        containing system, Pro, and Con messages.

    Returns:
        str: A formatted string with annotated Pro/Con messages.
    """
    annotated_history = []

    for msg in message_history:
        # Exclude system messages
        if msg.role == "system":
            continue

        # Map roles to Pro/Con
        if msg.role in ["pro", "user"]:  # User is treated as Pro in this context
            annotated_history.append(f"Pro: {msg.content}")
        elif msg.role in ["con", "assistant"]:  # Assistant is treated as Con
            annotated_history.append(f"Con: {msg.content}")

    return "\n".join(annotated_history)


def create_chat_agent(
    name: str, llm_config: OpenAIGPTConfig, system_message: str
) -> ChatAgent:
    """creates a ChatAgent with the given parameters.

    Args:
        name (str): The name of the agent.
        llm_config (OpenAIGPTConfig): The LLM configuration for the agent.
        system_message (str): The system message to guide the agent's LLM.

    Returns:
        ChatAgent: A configured ChatAgent instance.
    """
    return ChatAgent(
        ChatAgentConfig(
            llm=llm_config,
            name=name,
            system_message=system_message,
        )
    )


def run_debate() -> None:
    """Execute the main debate logic.

    Orchestrates the debate process, including setup, user input, LLM agent
    interactions, and final feedback. Handles both user-guided and LLM-
    delegated debates.

    This function:
    1. Loads global settings and the base LLM configurations.
    2. Prompts the user to confirm if they want to use same LLM for all agents.
    3. Prompts the user to select a debate topic and a side(Pro or Con).
    4. Sets up pro, con, and feedback agents.
    5. Runs the debate for a specified number of turns, either interactively
       or autonomously.
    6. Provides a feedback summary at the end.
    """

    global_settings = get_global_settings(nocache=True)
    lr.utils.configuration.set_global(global_settings)

    same_llm: bool = is_same_llm_for_all_agents()
    llm_delegate: bool = is_llm_delegate()
    max_turns: int = select_max_debate_turns()

    # Get base LLM configuration
    if same_llm:
        shared_agent_config: OpenAIGPTConfig = get_base_llm_config(
            select_model("main LLM")
        )
        pro_agent_config = con_agent_config = shared_agent_config

        # Create feedback_agent_config by modifying shared_agent_config
        feedback_agent_config: OpenAIGPTConfig = OpenAIGPTConfig(
            chat_model=shared_agent_config.chat_model,
            min_output_tokens=shared_agent_config.min_output_tokens,
            max_output_tokens=shared_agent_config.max_output_tokens,
            temperature=0.2,  # Override temperature
            seed=shared_agent_config.seed,
        )
        metaphor_search_agent_config = feedback_agent_config
    else:
        pro_agent_config: OpenAIGPTConfig = get_base_llm_config(
            select_model("for Pro Agent")
        )
        con_agent_config: OpenAIGPTConfig = get_base_llm_config(
            select_model("for Con Agent")
        )
        feedback_agent_config: OpenAIGPTConfig = get_base_llm_config(
            select_model("feedback"), temperature=0.2
        )
        metaphor_search_agent_config = feedback_agent_config

    system_messages: SystemMessages = load_system_messages(
        "examples/multi-agent-debate/system_messages.json"
    )
    topic_name, pro_key, con_key, side = select_topic_and_setup_side(system_messages)

    # Generate the system message
    metaphor_search_agent_system_message = (
        generate_metaphor_search_agent_system_message(system_messages, pro_key, con_key)
    )

    pro_agent = create_chat_agent(
        "Pro",
        pro_agent_config,
        system_messages.messages[pro_key].message + DEFAULT_SYSTEM_MESSAGE_ADDITION,
    )
    con_agent = create_chat_agent(
        "Con",
        con_agent_config,
        system_messages.messages[con_key].message + DEFAULT_SYSTEM_MESSAGE_ADDITION,
    )
    feedback_agent = create_chat_agent(
        "Feedback", feedback_agent_config, FEEDBACK_AGENT_SYSTEM_MESSAGE
    )
    metaphor_search_agent = MetaphorSearchChatAgent(  # Use the subclass here
        ChatAgentConfig(
            llm=metaphor_search_agent_config,
            name="MetaphorSearch",
            system_message=metaphor_search_agent_system_message,
        )
    )

    logger.info("Pro, Con, feedback, and metaphor_search agents created.")

    # Determine user's side and assign user_agent and ai_agent based on user selection
    agents = {
        "pro": (pro_agent, con_agent, "Pro", "Con"),
        "con": (con_agent, pro_agent, "Con", "Pro"),
    }
    user_agent, ai_agent, user_side, ai_side = agents[side]
    logger.info(
        f"Starting debate on topic: {topic_name}, taking the {user_side} side. "
        f"LLM Delegate: {llm_delegate}"
    )

    logger.info(f"\n{user_side} Agent ({topic_name}):\n")

    # Determine if the debate is autonomous or the user input for one side
    if llm_delegate:
        logger.info("Autonomous Debate Selected")
        interactive_setting = False
    else:
        logger.info("Manual Debate Selected with an AI Agent")
        interactive_setting = True
        user_input: str = Prompt.ask(
            "Your argument (or type 'f' for feedback, 'done' to end):"
        )
        user_agent.llm = None  # User message without LLM completion
        user_agent.user_message = user_input

    # Set up langroid tasks and run the debate
    user_task = Task(user_agent, interactive=interactive_setting, restart=False)
    ai_task = Task(ai_agent, interactive=False, single_round=True)
    user_task.add_sub_task(ai_task)
    if not llm_delegate:
        user_task.run(user_agent.user_message, turns=max_turns)
    else:
        user_task.run("get started", turns=max_turns)

    # Determine the last agent based on turn count and alternation
    # Note: user_agent and ai_agent are dynamically set based on the chosen user_side
    last_agent = ai_agent if max_turns % 2 == 0 else user_agent

    # Generate feedback summary and declare a winner using feedback agent
    if not last_agent.message_history:
        logger.warning("No agent message history found for the last agent")

    feedback_task = Task(feedback_agent, interactive=False, single_round=True)
    formatted_history = parse_and_format_message_history(last_agent.message_history)
    feedback_task.run(formatted_history)  # Pass formatted history to the feedback agent

    metaphor_search: bool = is_metaphor_search_key_set()

    if metaphor_search:
        metaphor_search_task = Task(metaphor_search_agent, interactive=False)
        metaphor_search_agent.enable_message(MetaphorSearchTool)
        metaphor_search_agent.enable_message(DoneTool)
        metaphor_search_task.run("run the search")

        url_docs_ask_questions = is_url_ask_question(topic_name)
        if url_docs_ask_questions:
            searched_urls = extract_urls(metaphor_search_agent.message_history)
            logger.info(searched_urls)
            ask_questions_agent = lr.agent.special.DocChatAgent(
                get_questions_agent_config(
                    searched_urls, feedback_agent_config.chat_model
                )
            )
            ask_questions_task = lr.Task(ask_questions_agent)
            ask_questions_task.run()


@app.command()
def main():
    """Main function and entry point for the Debate System"""
    run_debate()


if __name__ == "__main__":
    app()
