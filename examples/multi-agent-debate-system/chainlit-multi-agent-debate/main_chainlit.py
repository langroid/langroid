import chainlit as cl
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
)
import langroid.language_models as lm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
lr.utils.logging.setup_colored_logging()


# === SETUP AGENTS AND TASKS ===
def setup_agents_and_tasks(base_llm_config, debug=False, nocache=False):
    """
    Sets up agents and tasks for the debate.

    Args:
        base_llm_config: The base LLM configuration.
        debug (bool): Enable or disable debug mode.
        nocache (bool): Enable or disable caching.

    Returns:
        dict: A dictionary of tasks mapped by their names.
    """
    logger.info("Setting up agents and tasks...")
    global_settings = get_global_settings(debug, nocache)
    lr.utils.configuration.set_global(global_settings)
    logger.info("Global settings configured. Debug=%s, Cache=%s", debug, not nocache)

    logger.info("Base LLM configuration obtained.")

    # Create agents and reset memory for LLMS using the functions from agents.py
    # AI in Healthcare Pro and Con Agents
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


@cl.on_chat_start
async def main():
    logger.info("Application started.")
    cl.user_session.set('state', 'awaiting_llm_model')
    await cl.Message(content="Which OpenAI Model do you want to use? Select an option:\n1: GPT4o\n2: GPT4\n3: GPT3.5\n4: Mistral: mistral:7b-instruct-v0.2-q8_0\nEnter 1, 2, 3, or 4:").send()


@cl.on_message
async def handle_message(message):
    state = cl.user_session.get('state')
    if state == 'awaiting_llm_model':
        user_input = message.content.strip()
        chat_model_option = user_input
        model_map = {
            "1": "GPT4o",
            "2": "GPT4",
            "3": "GPT3.5"
        }

        if chat_model_option == "4":
            chat_model = "ollama/mistral:7b-instruct-v0.2-q8_0"
            base_llm_config = lm.OpenAIGPTConfig(
                chat_model=chat_model,
                chat_context_length=16000  # Only set for Ollama model
            )
        elif chat_model_option in model_map:
            chat_model = getattr(lm.OpenAIChatModel, model_map[chat_model_option])
            base_llm_config = lm.OpenAIGPTConfig(
                chat_model=chat_model
            )
        else:
            await cl.Message(content="Invalid option selected. Please enter 1, 2, 3, or 4.").send()
            return

        cl.user_session.set('base_llm_config', base_llm_config)
        logger.info("Base LLM configuration obtained.")

        tasks = setup_agents_and_tasks(base_llm_config)
        cl.user_session.set('tasks', tasks)

        cl.user_session.set('state', 'awaiting_delegate_setting')
        await cl.Message(content="Would you like the LLM to autonomously continue the debate without waiting for user input? (True/False)").send()

    elif state == 'awaiting_delegate_setting':
        user_input = message.content.strip().lower()
        llm_delegate_setting = user_input == 'true'
        cl.user_session.set('llm_delegate', llm_delegate_setting)
        logger.info("LLM Delegate setting: %s", llm_delegate_setting)

        cl.user_session.set('state', 'awaiting_debate_topic')
        await cl.Message(content="Select a debate topic:\n1. AI in Healthcare\n2. AI and Intellectual Property Rights\n3. AI and Societal Biases\n4. AI as an Educator").send()
    elif state == 'awaiting_debate_topic':
        user_input = message.content.strip()
        debate_topic = user_input
        logger.info("Debate topic selected: %s", debate_topic)
        cl.user_session.set('debate_topic', debate_topic)

        if debate_topic == "1":
            cl.user_session.set('state', 'awaiting_student_side')
            cl.user_session.set('debate_topic_name', 'AI in Healthcare')
            await cl.Message(content="Which side would you like to debate on?\n1. Pro-AI\n2. Con-AI").send()
        elif debate_topic == "2":
            cl.user_session.set('state', 'awaiting_student_side')
            cl.user_session.set('debate_topic_name', 'AI and Intellectual Property Rights')
            await cl.Message(content="Which side would you like to debate on?\n1. Pro-IP\n2. Con-IP").send()
        elif debate_topic == "3":
            cl.user_session.set('state', 'awaiting_student_side')
            cl.user_session.set('debate_topic_name', 'AI and Societal Biases')
            await cl.Message(content="Which side would you like to debate on?\n1. Pro-Bias\n2. Con-Bias").send()
        elif debate_topic == "4":
            cl.user_session.set('state', 'awaiting_student_side')
            cl.user_session.set('debate_topic_name', 'AI as an Educator')
            await cl.Message(content="Which side would you like to debate on?\n1. Pro-Edu\n2. Con-Edu").send()
        else:
            await cl.Message(content="Invalid topic selected. Please select 1, 2, 3, or 4.").send()
    elif state == 'awaiting_student_side':
        user_input = message.content.strip()
        student_side = user_input
        cl.user_session.set('student_side', student_side)
        logger.info("Student selected side: %s", student_side)

        debate_topic = cl.user_session.get('debate_topic')
        tasks = cl.user_session.get('tasks')

        if debate_topic == "1":
            selected_agent_task = tasks['pro_ai_task'] if student_side == "1" else tasks['con_ai_task']
            opposing_task = tasks['con_ai_task'] if student_side == "1" else tasks['pro_ai_task']
        elif debate_topic == "2":
            selected_agent_task = tasks['pro_ip_task'] if student_side == "1" else tasks['con_ip_task']
            opposing_task = tasks['con_ip_task'] if student_side == "1" else tasks['pro_ip_task']
        elif debate_topic == "3":
            selected_agent_task = tasks['pro_bias_task'] if student_side == "1" else tasks['con_bias_task']
            opposing_task = tasks['con_bias_task'] if student_side == "1" else tasks['pro_bias_task']
        elif debate_topic == "4":
            selected_agent_task = tasks['pro_edu_task'] if student_side == "1" else tasks['con_edu_task']
            opposing_task = tasks['con_edu_task'] if student_side == "1" else tasks['pro_edu_task']
        else:
            await cl.Message(content="Invalid selection. Please select 1 or 2.").send()
            return

        cl.user_session.set('selected_agent_task', selected_agent_task)
        cl.user_session.set('opposing_task', opposing_task)
        cl.user_session.set('student_arguments', [])
        cl.user_session.set('opposing_arguments', [])
        cl.user_session.set('is_pro_turn', True)
        cl.user_session.set('max_turns', 4)  # Maximum number of exchanges
        cl.user_session.set('current_turn', 0)
        cl.user_session.set('debate_ended_by_user', False)

        llm_delegate = cl.user_session.get('llm_delegate')
        if llm_delegate:
            # Start the autonomous debate
            await cl.Message(content="Starting autonomous debate...").send()
            await run_autonomous_debate()
        else:
            cl.user_session.set('state', 'debate_in_progress')
            await cl.Message(content="Please provide your argument or rebuttal. Type 'f' to request feedback or 'done' to end the debate.").send()
    elif state == 'debate_in_progress':
        user_input = message.content.strip()
        tasks = cl.user_session.get('tasks')
        if user_input.lower() == 'f':
            logger.info("Feedback requested.")
            student_arguments = cl.user_session.get('student_arguments')
            opposing_arguments = cl.user_session.get('opposing_arguments')
            feedback_content = "\n".join([
                "Student's Arguments:",
                *student_arguments,
                "\nOpposing Agent's Arguments:",
                *opposing_arguments,
            ])
            feedback_task = tasks['feedback_task']
            feedback_response = await cl.make_async(feedback_task.run)(feedback_content)
            feedback = feedback_response.content
            logger.info("Feedback provided.")
            await cl.Message(content="\nFeedback:\n" + feedback).send()
            await cl.Message(content="Please provide your next argument or rebuttal. Type 'f' to request feedback or 'done' to end the debate.").send()
        elif user_input.lower() == 'done':
            logger.info("Debate ended by student.")
            cl.user_session.set('debate_ended_by_user', True)
            await provide_feedback()
        else:
            if not user_input.strip():
                await cl.Message(content="No argument provided. Please provide your argument or type 'done' to end the debate.").send()
            else:
                # Store the student's argument
                student_arguments = cl.user_session.get('student_arguments')
                student_arguments.append(user_input)
                cl.user_session.set('student_arguments', student_arguments)
                logger.info("Student's argument added.")

                # Construct context for the opposing agent
                context = f"Opponent's argument: {user_input}"
                logger.info("Running opposing agent's response.")
                opposing_task = cl.user_session.get('opposing_task')
                response = await cl.make_async(opposing_task.run)(context)

                # Store the opposing agent's response
                if response:
                    opposing_arguments = cl.user_session.get('opposing_arguments')
                    opposing_arguments.append(response.content)
                    cl.user_session.set('opposing_arguments', opposing_arguments)
                    logger.info("Opposing agent's response added.")
                    await cl.Message(content="Opposing agent's response:\n" + response.content).send()

                # Check if max_turns reached
                current_turn = cl.user_session.get('current_turn', 0) + 1
                cl.user_session.set('current_turn', current_turn)
                max_turns = cl.user_session.get('max_turns')

                if current_turn >= max_turns:
                    await provide_feedback()
                else:
                    await cl.Message(content="Please provide your next argument or rebuttal. Type 'f' to request feedback or 'done' to end the debate.").send()
    else:
        await cl.Message(content="An error occurred. Please restart the conversation.").send()


async def run_autonomous_debate():
    tasks = cl.user_session.get('tasks')
    selected_agent_task = cl.user_session.get('selected_agent_task')
    opposing_task = cl.user_session.get('opposing_task')
    student_arguments = cl.user_session.get('student_arguments')
    opposing_arguments = cl.user_session.get('opposing_arguments')
    is_pro_turn = cl.user_session.get('is_pro_turn')
    max_turns = cl.user_session.get('max_turns')

    for turn in range(max_turns):
        last_argument = student_arguments[-1] if student_arguments else ""
        context = f"Opponent's argument: {last_argument}"
        if is_pro_turn:
            response = await cl.make_async(selected_agent_task.run)(context)
            if response:
                student_arguments.append(response.content)
                logger.info("Pro agent's response added.")
                await cl.Message(content="Pro agent's response:\n" + response.content).send()
        else:
            response = await cl.make_async(opposing_task.run)(context)
            if response:
                opposing_arguments.append(response.content)
                logger.info("Con agent's response added.")
                await cl.Message(content="Con agent's response:\n" + response.content).send()
        is_pro_turn = not is_pro_turn

    await provide_feedback()


async def provide_feedback():
    logger.info("Providing feedback.")
    student_arguments = cl.user_session.get('student_arguments')
    opposing_arguments = cl.user_session.get('opposing_arguments')
    feedback_content = "\n".join([
        "Student's Arguments:",
        *student_arguments,
        "\nOpposing Agent's Arguments:",
        *opposing_arguments,
    ])
    feedback_task = cl.user_session.get('tasks')['feedback_task']
    feedback_response = await cl.make_async(feedback_task.run)(feedback_content)
    feedback = feedback_response.content
    await cl.Message(content="\nFeedback:\n" + feedback).send()
    await cl.Message(content="Debate completed. Thank you for participating.").send()
    cl.user_session.set('state', 'debate_completed')

