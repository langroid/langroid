import typer
from rich.prompt import Prompt, Confirm
import json
from agents import create_agent
from config import get_base_llm_config, get_global_settings
from models import SystemMessages, Message
import logging
import langroid.utils.logging

# Initialize typer application
app = typer.Typer()

# Set up logging
logger = logging.getLogger(__name__)

# Suppress lower-level logs from langroid and other modules
logging.getLogger('langroid').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)


# Load and validate system messages from a JSON file
def load_system_messages(file_path: str) -> SystemMessages:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        # Map dictionaries to Message objects
        messages = {
            key: Message(**value) for key, value in data.items()
        }
        return SystemMessages(messages=messages)
    except Exception as e:
        logger.error(f"Error loading system messages: {e}")
        raise


# Prompt user to select a topic
def select_debate_topic():
    topics = [
        ("AI in Healthcare", "pro_ai", "con_ai"),
        ("AI and Intellectual Property", "pro_ip", "con_ip"),
        ("AI and Societal Biases", "pro_bias", "con_bias"),
        ("AI as an Educator", "pro_edu", "con_edu"),
    ]
    topic_choices = "\n".join([f"{i + 1}. {topic[0]}" for i, topic in enumerate(topics)])
    topic_index = int(
        Prompt.ask(
            f"Select a debate topic:\n{topic_choices}",
            choices=[str(i + 1) for i in range(len(topics))],
            default="1",
        )
    ) - 1
    selected_topic = topics[topic_index]
    logger.info(f"Selected topic: {selected_topic[0]}")
    return selected_topic


# Prompt user to select their side
def select_side(topic_name):
    side = Prompt.ask(
        f"Which side would you like to debate on?\n1. Pro-{topic_name}\n2. Con-{topic_name}",
        choices=["1", "2"],
        default="1",
    )
    return "pro" if side == "1" else "con"


# Prompt user to decide on LLM delegation
def is_llm_delegate():
    return Confirm.ask(
        "Would you like the LLM to autonomously continue the debate without waiting for user input?",
        default=False,
    )


# Main debate function
def run_debate():
    try:
        # Get global settings
        global_settings = get_global_settings(nocache=True)
        langroid.utils.configuration.set_global(global_settings)

        # Get base LLM configuration with the streaming handler
        agent_config = get_base_llm_config()
        system_messages = load_system_messages("system_messages.json")
        llm_delegate = is_llm_delegate()

        # Select topic and sides
        selected_topic_tuple = select_debate_topic()
        topic_name, pro_key, con_key = selected_topic_tuple
        side = select_side(topic_name)

        # Create agents for pro, con, and feedback agents.
        pro_agent = create_agent(
            agent_config, system_messages.messages[pro_key].message
        )
        con_agent = create_agent(
            agent_config, system_messages.messages[con_key].message
        )
        feedback_agent = create_agent(
            agent_config, system_messages.messages["feedback"].message
        )
        logger.info("Pro, Con, and feedback agents started")
        # Determine which agent the user is taking
        if side == "pro":
            user_agent = pro_agent
            ai_agent = con_agent
            user_side = "Pro"
            ai_side = "Con"
        else:
            user_agent = con_agent
            ai_agent = pro_agent
            user_side = "Con"
            ai_side = "Pro"

        logger.info(
            f"Starting debate on topic: {topic_name}, taking the {user_side} side. LLM Delegate: {llm_delegate}"
        )

        student_arguments = []
        ai_arguments = []
        max_turns = 4
        is_user_turn = True

        for turn in range(max_turns):
            if llm_delegate:
                current_agent = user_agent if is_user_turn else ai_agent
                agent_role = user_side if is_user_turn else ai_side

                # Build the conversation context
                if is_user_turn:
                    opponent_arguments = ai_arguments
                else:
                    opponent_arguments = student_arguments

                context = "\n".join(opponent_arguments[-1:]) if opponent_arguments else "Start of debate."

                # Prepare the full prompt
                full_context = (f"{current_agent.message_history}\n\n"
                                f"Please ensure the response include rebuttals to all Opponent's argument:\n{context}")

                # Call the agent without send_token_fn (handled in LLM config)
                print(f"\n{agent_role} Agent ({topic_name}):\n", end='', flush=True)
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
                    user_input = Prompt.ask("Your argument (or type 'f' for feedback, 'done' to end):")
                    if user_input.lower() == "f":
                        # Provide feedback during the debate
                        feedback_content = "\n".join(student_arguments + ai_arguments)
                        print("\nFeedback:\n", end='', flush=True)
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
                    context = student_arguments[-1] if student_arguments else "Start of debate."
                    # Include the AI agent's system message
                    full_context = f"{ai_agent.config.system_message}\n\nOpponent's argument:\n{context}"
                    print(f"\n{ai_side} Agent ({topic_name}):\n", end='', flush=True)
                    response = ai_agent.llm_response(full_context)
                    if response and response.content:
                        argument = response.content.strip()
                        ai_arguments.append(argument)
                    else:
                        print(f"\n{ai_side} Agent did not respond.")
                    is_user_turn = not is_user_turn

        # Final feedback
        final_feedback_content = "\n".join(student_arguments + ai_arguments)
        print("\nFinal Feedback:\n", end='', flush=True)
        final_feedback = feedback_agent.llm_response(
            f"Summarize the debate and declare a winner.\n{final_feedback_content}"
        )
        print()  # Newline after final feedback

    except Exception as e:
        logger.error(f"Unexpected error during debate: {e}")
        raise


@app.command()
def main():
    run_debate()


if __name__ == "__main__":
    app()
