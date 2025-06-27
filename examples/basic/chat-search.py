"""
This is a basic example of a chatbot that uses one of these web-search Tools to
answer questions:
 - GoogleSearchTool
 - DuckduckgoSearchTool
 - ExaSearchTool
When the LLM doesn't know the answer to a question, it will use the tool to
search the web for relevant results, and then use the results to answer the
question.

Run like this:

    python3 examples/basic/chat-search.py

or

    uv run examples/basic/chat-search.py -m groq/deepseek-r1-distill-llama-70b

There are optional args, especially note these:

-p or --provider: google or ddg or Exa (default: google)
-m <model_name>: to run with a different LLM model (default: gpt4-turbo)

You can specify a local in a few different ways, e.g. `-m local/localhost:8000/v1`
or `-m ollama/mistral` etc. See here how to use Langroid with local LLMs:
https://langroid.github.io/langroid/tutorials/local-llm-setup/


NOTE:
(a) If using Google Search, you must have GOOGLE_API_KEY and GOOGLE_CSE_ID
environment variables in your `.env` file, as explained in the
[README](https://github.com/langroid/langroid#gear-installation-and-setup).


(b) If using ExaSearchTool, you need to:
* set the EXA_API_KEY environment variables in
your `.env` file, e.g. `EXA_API_KEY=your_api_key_here`
* install langroid with the `exa` extra, e.g.
`pip install langroid[exa]` or `uv pip install langroid[exa]`
or `poetry add langroid[exa]`  or `uv add langroid[exa]`
(it installs the `exa-py` package from pypi).
For more information, please refer to the official docs:
https://exa.ai/

"""

import typer
from dotenv import load_dotenv
from rich import print

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.duckduckgo_search_tool import DuckduckgoSearchTool
from langroid.agent.tools.google_search_tool import GoogleSearchTool
from langroid.utils.configuration import Settings, set_global

app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    provider: str = typer.Option(
        "ddg",
        "--provider",
        "-p",
        help="search provider name (google, ddg, exa)",
    ),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    print(
        """
        [blue]Welcome to the Web Search chatbot!
        I will try to answer your questions, relying on (summaries of links from) 
        Web-Search when needed.
        
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        chat_context_length=32_000,
        temperature=0.15,
        max_output_tokens=1000,
        timeout=45,
    )

    match provider:
        case "google":
            search_tool_class = GoogleSearchTool
        case "exa":
            from langroid.agent.tools.exa_search_tool import ExaSearchTool

            search_tool_class = ExaSearchTool
        case "ddg":
            search_tool_class = DuckduckgoSearchTool
        case _:
            raise ValueError(f"Unsupported provider {provider} specified.")

    search_tool_handler_method = search_tool_class.name()
    config = lr.ChatAgentConfig(
        name="Seeker",
        handle_llm_no_tool="user",  # fwd to user when LLM sends non-tool msg
        llm=llm_config,
        vecdb=None,
        system_message=f"""
        You are a helpful assistant. You will try your best to answer my questions.
        Here is how you should answer my questions:
        - IF my question is about a topic you ARE CERTAIN about, answer it directly
        - OTHERWISE, use the `{search_tool_handler_method}` tool/function-call to
          get up to 5 results from a web-search, to help you answer the question.
          I will show you the results from the web-search, and you can use those
          to answer the question.
        - If I EXPLICITLY ask you to search the web/internet, then use the 
            `{search_tool_handler_method}` tool/function-call to get up to 5 results
            from a web-search, to help you answer the question.


        In case you use the TOOL `{search_tool_handler_method}`, you MUST WAIT
        for results from this tool; do not make up results!
        
        Be very CONCISE in your answers, use no more than 1-2 sentences.
        When you answer based on a web search, First show me your answer, 
        and then show me the SOURCE(s) and EXTRACT(s) to justify your answer,
        in this format:
        
        <your answer here>
        SOURCE: https://www.wikihow.com/Be-a-Good-Assistant-Manager
        EXTRACT: Be a Good Assistant ... requires good leadership skills.
        
        SOURCE: ...
        EXTRACT: ...
        
        For the EXTRACT, ONLY show up to first 3 words, and last 3 words.
        DO NOT MAKE UP YOUR OWN SOURCES; ONLY USE SOURCES YOU FIND FROM A WEB SEARCH.
        """,
    )
    agent = lr.ChatAgent(config)

    agent.enable_message(search_tool_class)

    task = lr.Task(agent, interactive=False)

    # local models do not like the first message to be empty
    user_message = "Can you help me with some questions?"
    task.run(user_message)


if __name__ == "__main__":
    app()
