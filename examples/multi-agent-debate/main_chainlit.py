import chainlit as cl
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
from config import get_global_settings
import langroid.language_models as lm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
lr.utils.logging.setup_colored_logging()

# Define the debate topics and corresponding agent keys
DEBATE_TOPICS = {
	"1": ("AI in Healthcare", "pro_ai", "con_ai"),
	"2": ("AI and Intellectual Property Rights", "pro_ip", "con_ip"),
	"3": ("AI and Societal Biases", "pro_bias", "con_bias"),
	"4": ("AI as an Educator", "pro_edu", "con_edu"),
}

def setup_agents_and_tasks(base_llm_config, debug=False, nocache=False):
	"""
	Sets up agents and tasks dynamically based on the debate topics.

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

	agents = {}
	tasks = {}

	for _, (topic_name, pro_key, con_key) in DEBATE_TOPICS.items():
		# Create agents dynamically
		agents[pro_key] = globals()[f"create_{pro_key}_agent"](base_llm_config)
		agents[con_key] = globals()[f"create_{con_key}_agent"](base_llm_config)

		# Create tasks for each agent
		tasks[f"{pro_key}_task"] = globals()[f"create_{pro_key}_task"](agents[pro_key])
		tasks[f"{con_key}_task"] = globals()[f"create_{con_key}_task"](agents[con_key])

		# Reset agent states
		for agent in [agents[pro_key], agents[con_key]]:
			agent.init_state()

	logger.info(f"Agents and tasks for {topic_name} created successfully.")

	# Add feedback agent and task
	feedback_agent = create_feedback_agent(base_llm_config)
	agent.init_state()
	tasks["feedback_task"] = create_feedback_task(feedback_agent)
	logger.info("Feedback agent and task created successfully.")

	logger.info("All agents and tasks have been set up.")
	return tasks

@cl.on_chat_start
async def main():
	logger.info("Application started.")
	cl.user_session.set('state', 'awaiting_llm_model')
	await cl.Message(
		content=(
			"Which model would you like to use? Select an option:\n"
			"1: GPT-4o\n"
			"2: GPT-4\n"
			"3: Mistral (mistral-7b-instruct)\n"
			"Enter 1, 2, or 3:"
		)
	).send()

@cl.on_message
async def handle_message(message):
	state = cl.user_session.get('state')

	if state == 'awaiting_llm_model':
		user_input = message.content.strip()
		if user_input == "1":
			chat_model = lm.OpenAIChatModel.GPT4o
			base_llm_config = lm.OpenAIGPTConfig(chat_model=chat_model)
		elif user_input == "2":
			chat_model = lm.OpenAIChatModel.GPT4
			base_llm_config = lm.OpenAIGPTConfig(chat_model=chat_model)
		elif user_input == "3":
			# Configure the Mistral model using Ollama
			chat_model = "ollama/mistral-7b-instruct"
			base_llm_config = lm.OllamaConfig(
				model=chat_model,
				chat_context_length=4096
			)
		else:
			await cl.Message(content="Invalid option selected. Please enter 1, 2, or 3.").send()
			return

		cl.user_session.set('base_llm_config', base_llm_config)
		logger.info("Base LLM configuration set with model: %s", chat_model if isinstance(chat_model, str) else chat_model.name)

		tasks = setup_agents_and_tasks(base_llm_config)
		cl.user_session.set('tasks', tasks)

		cl.user_session.set('state', 'awaiting_delegate_setting')
		await cl.Message(
			content="Would you like the LLM to autonomously continue the debate without waiting for user input? (True/False)"
		).send()

	elif state == 'awaiting_delegate_setting':
		user_input = message.content.strip().lower()
		if user_input in ['true', 'false']:
			llm_delegate_setting = user_input == 'true'
			cl.user_session.set('llm_delegate', llm_delegate_setting)
			logger.info("LLM Delegate setting: %s", llm_delegate_setting)

			cl.user_session.set('state', 'awaiting_debate_topic')
			topics_message = "\n".join([f"{key}. {value[0]}" for key, value in DEBATE_TOPICS.items()])
			await cl.Message(content=f"Select a debate topic:\n{topics_message}").send()
		else:
			await cl.Message(content="Invalid input. Please enter 'True' or 'False'.").send()

	elif state == 'awaiting_debate_topic':
		user_input = message.content.strip()
		topic_info = DEBATE_TOPICS.get(user_input)
		if topic_info:
			topic_name, pro_key, con_key = topic_info
			cl.user_session.set('debate_topic', topic_name)
			cl.user_session.set('pro_key', pro_key)
			cl.user_session.set('con_key', con_key)
			cl.user_session.set('state', 'awaiting_student_side')
			await cl.Message(
				content=f"Which side would you like to debate on?\n1. Pro-{topic_name}\n2. Con-{topic_name}"
			).send()
		else:
			await cl.Message(content="Invalid topic selected. Please select a valid option.").send()

	elif state == 'awaiting_student_side':
		user_input = message.content.strip()
		if user_input in ["1", "2"]:
			pro_key = cl.user_session.get('pro_key')
			con_key = cl.user_session.get('con_key')
			tasks = cl.user_session.get('tasks')
			selected_task = tasks[f"{pro_key}_task"] if user_input == "1" else tasks[f"{con_key}_task"]
			opposing_task = tasks[f"{con_key}_task"] if user_input == "1" else tasks[f"{pro_key}_task"]
			cl.user_session.set('selected_agent_task', selected_task)
			cl.user_session.set('opposing_task', opposing_task)
			cl.user_session.set('state', 'debate_in_progress')
			# Initialize debate variables
			cl.user_session.set('student_arguments', [])
			cl.user_session.set('opposing_arguments', [])
			cl.user_session.set('current_turn', 0)
			cl.user_session.set('max_turns', 4)
			llm_delegate = cl.user_session.get('llm_delegate')
			if llm_delegate:
				await cl.Message(content="Starting autonomous debate...").send()
				await run_autonomous_debate()
			else:
				await cl.Message(content="Debate started. Please provide your first argument.").send()
		else:
			await cl.Message(content="Invalid selection. Please select 1 or 2.").send()

	elif state == 'debate_in_progress':
		user_input = message.content.strip()
		if user_input.lower() == 'f':
			await provide_feedback()
		elif user_input.lower() == 'done':
			await provide_feedback(final=True)
		else:
			if not user_input:
				await cl.Message(content="No argument provided. Please provide your argument or type 'done' to end the debate.").send()
				return
			# Store the student's argument
			student_arguments = cl.user_session.get('student_arguments')
			student_arguments.append(user_input)
			cl.user_session.set('student_arguments', student_arguments)
			logger.info("Student's argument recorded.")

			# Get the opposing agent's response
			opposing_task = cl.user_session.get('opposing_task')
			response = await cl.make_async(opposing_task.run)(user_input)

			if response and response.content:
				opposing_arguments = cl.user_session.get('opposing_arguments')
				opposing_arguments.append(response.content.strip())
				cl.user_session.set('opposing_arguments', opposing_arguments)
				logger.info("Opposing agent's response recorded.")
				await cl.Message(content=f"Opposing Agent: {response.content.strip()}").send()

			# Check if maximum turns reached
			current_turn = cl.user_session.get('current_turn') + 1
			cl.user_session.set('current_turn', current_turn)
			max_turns = cl.user_session.get('max_turns')

			if current_turn >= max_turns:
				await provide_feedback(final=True)
			else:
				await cl.Message(content="Your turn to respond or type 'f' for feedback, 'done' to end:").send()
	else:
		await cl.Message(content="An error occurred. Please restart the conversation.").send()

async def run_autonomous_debate():
	selected_agent_task = cl.user_session.get('selected_agent_task')
	opposing_task = cl.user_session.get('opposing_task')
	student_arguments = cl.user_session.get('student_arguments')
	opposing_arguments = cl.user_session.get('opposing_arguments')
	max_turns = cl.user_session.get('max_turns')

	for turn in range(max_turns):
		# Student's turn
		context = opposing_arguments[-1] if opposing_arguments else "Start of debate."
		response = await cl.make_async(selected_agent_task.run)(context)
		if response and response.content:
			student_arguments.append(response.content.strip())
			logger.info(f"Pro Agent's response at turn {turn+1} recorded.")
			await cl.Message(content=f"Pro Agent: {response.content.strip()}").send()

		# Opponent's turn
		context = student_arguments[-1]
		response = await cl.make_async(opposing_task.run)(context)
		if response and response.content:
			opposing_arguments.append(response.content.strip())
			logger.info(f"Con Agent's response at turn {turn+1} recorded.")
			await cl.Message(content=f"Con Agent: {response.content.strip()}").send()

	await provide_feedback(final=True)

async def provide_feedback(final=False):
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
	if final:
		await cl.Message(content="Debate completed. Thank you for participating.").send()
		cl.user_session.set('state', 'debate_completed')
		logger.info("Debate completed and feedback provided.")
	else:
		await cl.Message(content="Please provide your next argument or type 'done' to end the debate.").send()
		logger.info("Feedback provided during debate.")

