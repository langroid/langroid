import typer
from rich.prompt import Prompt
import langroid as lr
from agents import (
    create_pro_ai_agent,
    create_con_ai_agent,
    create_pro_ip_agent,
    create_con_ip_agent,
    create_pro_bias_agent,
    create_con_bias_agent,
    create_pro_edu_agent,
    create_con_edu_agent,
    create_feedback_agent,
)
from tasks import (
    create_pro_ai_task,
    create_con_ai_task,
    create_pro_ip_task,
    create_con_ip_task,
    create_pro_bias_task,
    create_con_bias_task,
    create_pro_edu_task,
    create_con_edu_task,
    create_feedback_task,
)
from config import (
    get_global_settings,
    get_base_llm_config,
    get_llm_delegate_setting,
)
import logging

# Set up the Typer application
app = typer.Typer()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
lr.utils.logging.setup_colored_logging()



# === SETUP AGENTS AND TASKS ===
def setup_agents_and_tasks(debug=False, nocache=False):
    """
    Sets up agents and tasks for the debate.

    Args:
        debug (bool): Enable or disable debug mode.
        nocache (bool): Enable or disable caching.

    Returns:
        dict: A dictionary of tasks mapped by their names.
    """
    logger.info("Setting up agents and tasks...")
    global_settings = get_global_settings(debug, nocache)
    lr.utils.configuration.set_global(global_settings)
    logger.info("Global settings configured. Debug=%s, Cache=%s", debug, not nocache)

    base_llm_config = get_base_llm_config()
    logger.info("Base LLM configuration obtained.")


    # Create agents and reset memory for LLMS using the functions from agents.py
    #AI in Healthcare Pro and Con Agents
    pro_ai_agent = create_pro_ai_agent(base_llm_config)
    pro_ai_agent.clear_history(0)
    pro_ai_agent.dialog.clear()
    con_ai_agent = create_con_ai_agent(base_llm_config)
    con_ai_agent.clear_history(0)
    con_ai_agent.dialog.clear()
    logger.info("AI and Healthcare Pro and Con Agents created")

    # AI and Intellectual Property Pro and Con Agents
    pro_ip_agent = create_pro_ip_agent(base_llm_config)
    pro_ip_agent.clear_history(0)
    pro_ip_agent.dialog.clear()
    con_ip_agent = create_con_ip_agent(base_llm_config)
    con_ip_agent.clear_history(0)
    con_ip_agent.dialog.clear()
    logger.info("AI and IP Pro and Con Agents created")

    # AI and BIAS Pro and Con Agents
    pro_bias_agent = create_pro_bias_agent(base_llm_config)
    pro_bias_agent.clear_history(0)
    pro_bias_agent.dialog.clear()
    con_bias_agent = create_con_bias_agent(base_llm_config)
    con_bias_agent.clear_history(0)
    con_bias_agent.dialog.clear()
    logger.info("AI and Bias Pro and Con Agents created")

    # AI and Education Pro and Con Agents
    pro_edu_agent = create_pro_edu_agent(base_llm_config)
    pro_edu_agent.clear_history(0)
    pro_edu_agent.dialog.clear()
    con_edu_agent = create_con_edu_agent(base_llm_config)
    con_edu_agent.clear_history(0)
    con_edu_agent.dialog.clear()
    logger.info("AI and Education Pro and Con Agents created")

    # Feedback Agent
    feedback_agent = create_feedback_agent(base_llm_config)
    feedback_agent.clear_history(0)
    feedback_agent.dialog.clear()
    logger.info("Feedback Agent created")

    logger.info("All Agents created successfully.")



    # Create tasks for each agent using functions from tasks.py
    tasks = {
        'pro_ai_task': create_pro_ai_task(pro_ai_agent),
        'con_ai_task': create_con_ai_task(con_ai_agent),
        'pro_ip_task': create_pro_ip_task(pro_ip_agent),
        'con_ip_task': create_con_ip_task(con_ip_agent),
        'pro_bias_task': create_pro_bias_task(pro_bias_agent),
        'con_bias_task': create_con_bias_task(con_bias_agent),
        'pro_edu_task': create_pro_edu_task(pro_edu_agent),
        'con_edu_task': create_con_edu_task(con_edu_agent),
        'feedback_task': create_feedback_task(feedback_agent),
    }

    logger.info("Tasks created successfully.")
    return tasks


# === RUN THE DEBATE LOOP ===
def run_debate(tasks):
    """
    Runs the debate interaction logic.

    Args:
        tasks (dict): A dictionary of available debate tasks.
    """

    logger.info("Starting debate session.")

    # Prompt for LLM delegate setting
    llm_delegate = get_llm_delegate_setting()
    logger.info("LLM Delegate setting: %s", llm_delegate)

    # === Select Debate Topic ===
    debate_topic = Prompt.ask(
        "Select a debate topic:\n1. AI in Healthcare\n2. AI and Intellectual Property Rights\n"
        "3. AI and Societal Biases\n4. AI as an Educator",
        choices=["1", "2", "3", "4"],
        default="1",
    )
    logger.info("Debate topic selected: %s", debate_topic)

    # Selecting the topic and the appropriate Pro or Con agent
    if debate_topic == "1":
        student_side = Prompt.ask(
            "Which side would you like to debate on?\n1. Pro-AI\n2. Con-AI",
            choices=["1", "2"],
            default="1",
        )
        selected_agent_task = tasks['pro_ai_task'] if student_side == "1" else tasks['con_ai_task']
        opposing_task = tasks['con_ai_task'] if student_side == "1" else tasks['pro_ai_task']
    elif debate_topic == "2":
        student_side = Prompt.ask(
            "Which side would you like to debate on?\n1. Pro-IP\n2. Con-IP",
            choices=["1", "2"],
            default="1",
        )
        selected_agent_task = tasks['pro_ip_task'] if student_side == "1" else tasks['con_ip_task']
        opposing_task = tasks['con_ip_task'] if student_side == "1" else tasks['pro_ip_task']
    elif debate_topic == "3":
        student_side = Prompt.ask(
            "Which side would you like to debate on?\n1. Pro-Bias\n2. Con-Bias",
            choices=["1", "2"],
            default="1",
        )
        selected_agent_task = tasks['pro_bias_task'] if student_side == "1" else tasks['con_bias_task']
        opposing_task = tasks['con_bias_task'] if student_side == "1" else tasks['pro_bias_task']
    elif debate_topic == "4":
        student_side = Prompt.ask(
            "Which side would you like to debate on?\n1. Pro-Edu\n2. Con-Edu",
            choices=["1", "2"],
            default="1",
        )
        selected_agent_task = tasks['pro_edu_task'] if student_side == "1" else tasks['con_edu_task']
        opposing_task = tasks['con_edu_task'] if student_side == "1" else tasks['pro_edu_task']
    else:
        logger.error("Invalid topic selected.")
        return

    logger.info("Student selected side: %s", "Pro" if student_side == "1" else "Con")
    student_arguments = []
    opposing_arguments = []
    first_turn = True
    is_pro_turn = True
    max_turns = 4  # Maximum number of exchanges

    # === Debate Loop ===
    debate_ended_by_user = False  # Flag to track if the user ended the debate

    for turn in range(max_turns):
        if llm_delegate:
            # Autonomous debate mode
            last_argument = student_arguments[-1] if student_arguments else ""
            context = f"Opponent's argument: {last_argument}"
            if is_pro_turn:
                response = selected_agent_task.run(context)  # Pass context directly
                if response:
                    student_arguments.append(response.content)
                    logger.info("Pro agent's response added.")
                    print("Pro agent's response added.")
            else:
                response = opposing_task.run(context)  # Pass context directly
                if response:
                    opposing_arguments.append(response.content)
                    logger.info("Con agent's response added.")
                    print("Con agent's response added.")
            is_pro_turn = not is_pro_turn
        else:
            # Manual debate mode
            student_argument = Prompt.ask(
                "\nPlease provide your argument or rebuttal. Type 'f' to request feedback or 'done' to end the debate."
            )
            if student_argument.lower() == 'f':
                logger.info("Feedback requested.")
                feedback_content = "\n".join([
                    "Student's Arguments:",
                    *student_arguments,
                    "\nOpposing Agent's Arguments:",
                    *opposing_arguments,
                ])
                feedback = tasks['feedback_task'].run(feedback_content).content
                logger.info("Feedback provided.")
                print("\nFeedback:")
                print(feedback)
            elif student_argument.lower() == 'done':
                logger.info("Debate ended by student.")
                debate_ended_by_user = True  # Mark that the user ended the debate
                break
            else:
                if not student_argument.strip():
                    logger.warning("No argument provided.")
                    if student_arguments:
                        last_student_argument = student_arguments[-1]
                        logger.info("Using the last student argument for the opposing agent.")
                    else:
                        logger.warning("No previous argument available. Skipping turn.")
                        print("No previous argument to use. Skipping turn.")
                        continue
                else:
                    # Store the student's argument
                    student_arguments.append(student_argument)
                    logger.info("Student's argument added.")
                    print("Student's argument added.")

                    # Construct context for the opposing agent
                    context = f"Opponent's argument: {student_argument}"
                    logger.info("Running opposing agent's response.")
                    response = opposing_task.run(context)

                    # Store the opposing agent's response
                    if response:
                        opposing_arguments.append(response.content)
                        logger.info("Opposing agent's response added.")
                        print("Opposing agent's response added.")
            first_turn = False
            is_pro_turn = not is_pro_turn  # Alternate turn after each response

    # === Call Feedback Agent if the debate didn't end by user ===
    if not debate_ended_by_user:
        logger.info("Maximum turns reached or debate ended. Requesting feedback.")
        feedback_content = "\n".join([
            "Student's Arguments:",
            *student_arguments,
            "\nOpposing Agent's Arguments:",
            *opposing_arguments,
        ])
        feedback = tasks['feedback_task'].run(feedback_content).content
        print("\nFeedback:")
        print(feedback)

    logger.info("Debate completed. Exiting.")
    quit()


# === MAIN ENTRY POINT ===
@app.command()
def main(debug: bool = typer.Option(False, "--debug", "-d", help="Debug mode"),
         nocache: bool = typer.Option(False, "--nocache", "-nc", help="Don't use cache")) -> None:
    """
    Main function to initialize and run the debate.
    """
    logger.info("Application started. Debug=%s, NoCache=%s", debug, nocache)

    tasks = setup_agents_and_tasks(debug, nocache)
    run_debate(tasks)


if __name__ == "__main__":
    app()

