"""
Streamlit web app using a local LLM, with ollama.

This script makes use of the same logic used in examples/basic/fn-call-local-simple.py but using a streamlit app.

# (1) You should run pip install streamlit to ensure you have the library prior to running this app.

# (2) Run like this:

streamlit run examples/basic/fn-call-local-simple-with-streamlit.py

"""
import os
import streamlit as st
from pydantic import BaseModel, Field
import langroid as lr
from langroid.agent.tool_message import ToolMessage
from langroid.language_models import OpenAIGPTConfig
from langroid.agent.chat_document import ChatDocument
from typing import List

# Set environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Langroid LLM configuration
llm_cfg = OpenAIGPTConfig(
    chat_model="litellm/ollama/mistral:7b-instruct-v0.2-q4_K_M",
    chat_context_length=4096,
    max_output_tokens=100,
    temperature=0.2,
    stream=True,
    timeout=45,
)


# Pydantic models for City information
class CityData(BaseModel):
    population: int = Field(..., description="population of city")
    country: str = Field(..., description="country of city")


class City(BaseModel):
    name: str = Field(..., description="name of city")
    details: CityData = Field(..., description="details of city")


# CityTool class for the LLM
class CityTool(ToolMessage):
    """Present information about a city"""

    request: str = "city_tool"
    purpose: str = "present <city_info>"
    city_info: City = Field(..., description="information about a city")

    def handle(self) -> str:
        """Handle LLM's structured output if it matches City structure"""
        print("SUCCESS! Got Valid City Info")
        return """
        Thanks! ask me for another city name, do not say anything else
        until you get a city name.
        """

    @staticmethod
    def handle_message_fallback(
        agent: lr.ChatAgent, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """Fallback method when LLM forgets to generate a tool"""
        if isinstance(msg, ChatDocument) and msg.metadata.sender == "LLM":
            return """
            You must use the `city_tool` to generate city information.
            You either forgot to use it, or you used it with the wrong format.
            Make sure all fields are filled out.
            """

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        # Used to provide few-shot examples in the system prompt
        return [
            cls(
                city_info=City(
                    name="San Francisco",
                    details=CityData(
                        population=800_000,
                        country="USA",
                    ),
                )
            )
        ]


# ChatAgent and Task setup
config = lr.ChatAgentConfig(
    llm=llm_cfg,
    system_message="""
    You are an expert on world city information. 
    The user will give you a city name, and you should use the `city_tool` to
    generate information about the city, and present it to the user.
    Make up the values if you don't know them exactly, but make sure
    the structure is as specified in the `city_tool` JSON definition.

    DO NOT SAY ANYTHING ELSE BESIDES PROVIDING THE CITY INFORMATION.

    START BY ASKING ME TO GIVE YOU A CITY NAME. 
    DO NOT GENERATE ANYTHING YOU GET A CITY NAME.

    Once you've generated the city information using `city_tool`,
    ask for another city name, and so on.
    """,
)

agent = lr.ChatAgent(config)
agent.enable_message(CityTool)

# Streamlit app setup
st.title("City Information ChatBot 🏙️")

# Initialize session state for Streamlit
if "history" not in st.session_state:
    st.session_state["history"] = []


# Function to handle conversation
def conversation_chat(query):
    response = agent.llm_response(query)
    st.session_state["history"].append((query, response))
    return response


# UI for the chat
def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input(
                "City Name:", placeholder="Enter a city name", key="input"
            )
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            output = conversation_chat(user_input)
            st.session_state["history"].append((user_input, output))

    with reply_container:
        for i in range(len(st.session_state["history"])):
            question, answer = st.session_state["history"][i]
            st.write("You: " + question)
            st.write("Bot: " + str(answer))


# Display chat history
display_chat_history()
