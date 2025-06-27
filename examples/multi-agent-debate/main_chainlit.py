import logging
import os
from textwrap import dedent

import chainlit as cl
from chainlit_utils import (
    is_llm_delegate,
    is_metaphor_search_key_set,
    is_same_llm_for_all_agents,
    is_url_ask_question,
    select_max_debate_turns,
    select_model,
    select_topic_and_setup_side,
)
from config import get_base_llm_config, get_global_settings, get_questions_agent_config
from main import (
    MetaphorSearchChatAgent,
    create_chat_agent,
    parse_and_format_message_history,
)
from models import SystemMessages, load_system_messages
from system_messages import (
    DEFAULT_SYSTEM_MESSAGE_ADDITION,
    FEEDBACK_AGENT_SYSTEM_MESSAGE,
    generate_metaphor_search_agent_system_message,
)

# Import from utils.py
from utils import (
    extract_urls,
)

import langroid as lr
from langroid.agent.callbacks.chainlit import (
    ChainlitCallbackConfig,
    ChainlitTaskCallbacks,
    add_instructions,
)
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tools.metaphor_search_tool import MetaphorSearchTool
from langroid.agent.tools.orchestration import DoneTool
from langroid.language_models import OpenAIGPTConfig
from langroid.utils.configuration import settings
from langroid.utils.logging import setup_logger


class CustomChainlitTaskCallbacks(ChainlitTaskCallbacks):
    """
    Custom subclass of ChainlitTaskCallbacks with adjusted behavior for task integration.
    """

    def __init__(
        self,
        task: lr.Task,
        config: ChainlitCallbackConfig = ChainlitCallbackConfig(),
    ):
        """
        Initialize the custom task callbacks and recursively inject them.
        """
        # Pass the task directly instead of task.agent
        super().__init__(task, config)
        # Inject callbacks recursively
        self._inject_callbacks(task)
        self.task = task
        if config.show_subtask_response:
            self.task.callbacks.show_subtask_response = self.show_subtask_response

    def show_subtask_response(
        self, task: lr.Task, content: str, is_tool: bool = False
    ) -> None:
        """
        Override the display format for subtask responses.
        """

    @classmethod
    def _inject_callbacks(
        cls, task: lr.Task, config: ChainlitCallbackConfig = ChainlitCallbackConfig()
    ) -> None:
        """
        Recursively apply CustomChainlitTaskCallbacks to agents of sub-tasks.
        """
        for sub_task in task.sub_tasks:
            CustomChainlitTaskCallbacks(sub_task, config=config)


def create_custom_chat_agent(
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
    # Modify the system message to include instructions for the agent
    additional_system_message = """**Response format (strictly follow this structure):**  
    Pro:  
    - [First key point]  
    - [Second key point]  
    - [Third key point]
    **Limit responses to exactly 3 points expressed as single sentences.**"
    """
    system_message = f"""
       Start your response with '{name}: ' and then follow the instructions below.
        {system_message} {additional_system_message}
        """
    return ChatAgent(
        ChatAgentConfig(
            llm=llm_config,
            name=name,
            system_message=system_message,
        )
    )


@cl.on_chat_start
async def on_chat_start(
    debug: bool = os.getenv("DEBUG", False),
    no_cache: bool = os.getenv("NOCACHE", False),
):
    settings.debug = debug
    settings.cache = not no_cache

    # set info logger
    logger = setup_logger(__name__, level=logging.INFO, terminal=True)
    logger.info("Starting multi-agent-debate")

    await add_instructions(
        title="AI Powered Debate Platform",
        content=dedent(
            """
            Welcome to the Debate Platform.
            Interaction
            1. Decide if you want to you use same LLM for all agents or different ones
            2. Decide if you want autonomous debate between AI Agents or user vs. AI Agent. 
            3. Select a debate topic.
            4. Choose your side (Pro or Con).
            5. Engage in a debate by providing arguments and receiving responses from agents.
            6. Request feedback at any time by typing `f`.
            7. Decide if you want the Metaphor Search to run to find Topic relevant web links
            and summarize them. 
            8. Decide if you want to chat with the documents extracted from URLs found to learn more about the Topic.
            9. End the debate manually by typing "done". If you decide to chat with the documents, you can end session
            by typing "x"
            """
        ),
    )

    global_settings = get_global_settings(nocache=True)
    lr.utils.configuration.set_global(global_settings)

    same_llm = await is_same_llm_for_all_agents()
    llm_delegate: bool = await is_llm_delegate()
    max_turns: int = await select_max_debate_turns()
    print(max_turns)

    # Get base LLM configuration
    if same_llm:
        shared_agent_config: OpenAIGPTConfig = get_base_llm_config(
            await select_model("main LLM")
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
            await select_model("for Pro Agent")
        )
        con_agent_config: OpenAIGPTConfig = get_base_llm_config(
            await select_model("for Con Agent")
        )
        feedback_agent_config: OpenAIGPTConfig = get_base_llm_config(
            await select_model("feedback"), temperature=0.2
        )
        metaphor_search_agent_config = feedback_agent_config

    system_messages: SystemMessages = load_system_messages(
        "examples/multi-agent-debate/system_messages.json"
    )
    LLM_DELEGATE_FLAG: bool = llm_delegate

    topic_name, pro_key, con_key, side = await select_topic_and_setup_side(
        LLM_DELEGATE_FLAG, system_messages
    )

    # Generate the system message
    metaphor_search_agent_system_message = (
        generate_metaphor_search_agent_system_message(system_messages, pro_key, con_key)
    )
    # pro_agent_system_message = "You are Pro. Start your response with 'Pro: "
    # + system_messages.messages[pro_key].message + DEFAULT_SYSTEM_MESSAGE_ADDITION
    pro_agent = create_custom_chat_agent(
        "Pro",
        pro_agent_config,
        system_messages.messages[pro_key].message + DEFAULT_SYSTEM_MESSAGE_ADDITION,
    )
    con_agent = create_custom_chat_agent(
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
    if LLM_DELEGATE_FLAG:
        logger.info("Autonomous Debate Selected")
        interactive_setting = False
    else:
        logger.info("Manual Debate Selected with an AI Agent")
        interactive_setting = True
        user_input: str
        try:
            user_input_response = await cl.AskUserMessage(
                content="Your argument (or type 'f' for feedback, 'done' to end):",
                timeout=600,  # 10 minutes
            ).send()

            logger.info(f"Received user input response: {user_input_response}")

            if user_input_response and "output" in user_input_response:
                user_input = str(user_input_response["output"]).strip()
                logger.info(f"User input processed successfully: {user_input}")
                user_agent.llm = None  # User message without LLM completion
                user_agent.user_message = user_input
            else:
                logger.error("Response received but 'output' key is missing or empty.")
                raise TimeoutError(
                    "No valid response received for the user input question."
                )

        except TimeoutError as e:
            logger.error(str(e))
            # Handle timeout or invalid response gracefully

        # Assign the input to the user agent's attributes
        user_agent.llm = None  # User message without LLM completion
        user_agent.user_message = user_input

    # Set up langroid tasks and run the debate
    user_task = Task(user_agent, interactive=interactive_setting, restart=False)
    ai_task = Task(ai_agent, interactive=False, single_round=True)

    user_task.add_sub_task(ai_task)

    if not llm_delegate:
        ChainlitTaskCallbacks(user_task)
        await user_task.run_async(user_agent.user_message, turns=max_turns)
    else:
        CustomChainlitTaskCallbacks(user_task)
        await user_task.run_async("get started", turns=max_turns)

    # Determine the last agent based on turn count and alternation
    # Note: user_agent and ai_agent are dynamically set based on the chosen user_side
    last_agent = ai_agent if max_turns % 2 == 0 else user_agent

    await cl.Message(content="## Feedback and Debate Evaluation:").send()
    # Generate feedback summary and declare a winner using feedback agent

    if not last_agent.message_history:
        logger.warning("No agent message history found for the last agent")

    feedback_task = Task(
        feedback_agent,
        system_message=FEEDBACK_AGENT_SYSTEM_MESSAGE,
        interactive=False,
        single_round=True,
    )
    formatted_history = parse_and_format_message_history(last_agent.message_history)
    CustomChainlitTaskCallbacks(feedback_task)
    await feedback_task.run_async(
        formatted_history
    )  # Pass formatted history to the feedback agent

    metaphor_search: bool = await is_metaphor_search_key_set()

    if metaphor_search:
        metaphor_search_task = Task(
            metaphor_search_agent,
            system_message=metaphor_search_agent_system_message,
            interactive=False,
        )
        metaphor_search_agent.enable_message(MetaphorSearchTool)
        metaphor_search_agent.enable_message(DoneTool)
        CustomChainlitTaskCallbacks(metaphor_search_task)
        await metaphor_search_task.run_async("run the search")

        url_docs_ask_questions = await is_url_ask_question(topic_name)
        if url_docs_ask_questions:
            searched_urls = extract_urls(metaphor_search_agent.message_history)
            logger.info(searched_urls)
            ask_questions_agent = lr.agent.special.DocChatAgent(
                get_questions_agent_config(
                    searched_urls, feedback_agent_config.chat_model
                )
            )
            ask_questions_task = lr.Task(ask_questions_agent)
            CustomChainlitTaskCallbacks(ask_questions_task)
            await ask_questions_task.run_async()
