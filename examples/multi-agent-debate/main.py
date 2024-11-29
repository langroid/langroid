import typer
from rich.prompt import Prompt
import langroid as lr
from agents import (
    create_pro_ai_agent, create_con_ai_agent,
    create_pro_ip_agent, create_con_ip_agent,
    create_pro_bias_agent, create_con_bias_agent,
    create_pro_edu_agent, create_con_edu_agent,
    create_feedback_agent,
)
from tasks import (
    create_pro_ai_task, create_con_ai_task,
    create_pro_ip_task, create_con_ip_task,
    create_pro_bias_task, create_con_bias_task,
    create_pro_edu_task, create_con_edu_task,
    create_feedback_task,
)
from config import get_global_settings, get_base_llm_config, is_llm_delegate
import logging

# Set up the Typer application
app = typer.Typer()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
lr.utils.logging.setup_colored_logging()

# Mappings for dynamic task and agent creation
DEBATE_TOPICS = {
    "1": ("AI in Healthcare", "pro_ai", "con_ai"),
    "2": ("AI and Intellectual Property", "pro_ip", "con_ip"),
    "3": ("AI and Societal Biases", "pro_bias", "con_bias"),
    "4": ("AI as an Educator", "pro_edu", "con_edu"),
}


def setup_agents_and_tasks(debug=False, nocache=False):
    """Sets up agents and tasks dynamically."""
    logger.info("Setting up agents and tasks...")
    global_settings = get_global_settings(debug, nocache)
    lr.utils.configuration.set_global(global_settings)
    base_llm_config = get_base_llm_config()

    agents = {}
    tasks = {}

    for _, (topic_name, pro_key, con_key) in DEBATE_TOPICS.items():
        agents[pro_key] = globals()[f"create_{pro_key}_agent"](base_llm_config)
        agents[con_key] = globals()[f"create_{con_key}_agent"](base_llm_config)
        tasks[f"{pro_key}_task"] = globals()[f"create_{pro_key}_task"](agents[pro_key])
        tasks[f"{con_key}_task"] = globals()[f"create_{con_key}_task"](agents[con_key])

        # Reset agent states
        for agent in [agents[pro_key], agents[con_key]]:
            agent.init_state()

    # Add feedback agent and task
    feedback_agent = create_feedback_agent(base_llm_config)
    tasks["feedback_task"] = create_feedback_task(feedback_agent)

    logger.info("Agents and tasks successfully created.")
    return tasks


def select_debate_topic():
    """Prompts the user to select a debate topic."""
    topic_choices = "\n".join([f"{key}. {value[0]}" for key, value in DEBATE_TOPICS.items()])
    topic_key = Prompt.ask(f"Select a debate topic:\n{topic_choices}", choices=DEBATE_TOPICS.keys(), default="1")
    return DEBATE_TOPICS[topic_key]


def select_side(topic_name, pro_key, con_key, tasks):
    """Prompts the user to select a side of the debate."""
    side = Prompt.ask(
        f"Which side would you like to debate on?\n1. Pro-{topic_name}\n2. Con-{topic_name}",
        choices=["1", "2"],
        default="1",
    )
    return (
        tasks[f"{pro_key}_task"] if side == "1" else tasks[f"{con_key}_task"],
        tasks[f"{con_key}_task"] if side == "1" else tasks[f"{pro_key}_task"],
    )


def run_debate(tasks):
    """Runs the debate interaction logic."""
    llm_delegate = is_llm_delegate()
    topic_name, pro_key, con_key = select_debate_topic()
    selected_task, opposing_task = select_side(topic_name, pro_key, con_key, tasks)

    logger.info("Starting debate on topic: %s", topic_name)
    student_arguments = []
    opposing_arguments = []
    max_turns = 4
    is_pro_turn = True

    for turn in range(max_turns):
        if llm_delegate:
            context = student_arguments[-1] if student_arguments else "Start of debate."
            current_task = selected_task if is_pro_turn else opposing_task
            response = current_task.run(context)

            if response and response.content:
                argument = response.content.strip()
                (student_arguments if is_pro_turn else opposing_arguments).append(argument)
                print(f"\n{'Pro' if is_pro_turn else 'Con'} Agent: {argument}")
            else:
                print(f"\n{'Pro' if is_pro_turn else 'Con'} Agent did not respond.")
            is_pro_turn = not is_pro_turn
        else:
            user_input = Prompt.ask("Your argument (or type 'f' for feedback, 'done' to end):")
            if user_input.lower() == "f":
                feedback = tasks["feedback_task"].run("\n".join(student_arguments + opposing_arguments)).content
                print("\nFeedback:", feedback)
            elif user_input.lower() == "done":
                break
            else:
                student_arguments.append(user_input)
                response = opposing_task.run(user_input)
                if response and response.content:
                    opposing_arguments.append(response.content.strip())
                    print("\nCon Agent:", response.content.strip())

    final_feedback = tasks["feedback_task"].run("\n".join(student_arguments + opposing_arguments)).content
    print("\nFinal Feedback:", final_feedback)


@app.command()
def main(debug: bool = False, nocache: bool = False):
    tasks = setup_agents_and_tasks(debug, nocache)
    run_debate(tasks)


if __name__ == "__main__":
    app()

