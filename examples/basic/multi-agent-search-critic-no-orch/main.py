"""
Version of main.py, but does NOT use any inter-agent orchestration,
 i.e. we create a separate Task object from each agent, but we do not
 connect them as sub-tasks.
 Instead we write extra code to handle each task's output, and
 determine what to do with it.

3-Agent system where:
- Assistant takes user's (complex) question, breaks it down into smaller pieces
    if needed
- Searcher takes Assistant's question, uses the Search tool to search the web
    (using DuckDuckGo), and returns a coherent answer to the Assistant.
- Critic takes Assistant's final answer, and provides feedback on it.

Once the Assistant thinks it has enough info to answer the user's question, it
says DONE and presents the answer to the user.

See also: chat-search for a basic single-agent search

Run like this from root of repo:

python3 -m examples.basic.multi-agent-search-critic-no-orch.main

There are optional args, especially note these:

-m <model_name>: to run with a different LLM model (default: gpt4o)

For example try this question:

did Bach make more music than Beethoven?

You can specify a local LLM in a few different ways, e.g. `-m local/localhost:8000/v1`
or `-m ollama/mistral` etc. See here how to use Langroid with local LLMs:
https://langroid.github.io/langroid/tutorials/local-llm-setup/


"""

import typer
from dotenv import load_dotenv
from rich import print
from rich.prompt import Prompt

import langroid as lr
from langroid.utils.configuration import Settings, set_global

from .assistant_agent import make_assistant_task
from .critic_agent import make_critic_task
from .search_agent import make_search_task
from .tools import AnswerTool, FeedbackTool, FinalAnswerTool, QuestionTool

app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )
    print(
        """
        [blue]Welcome to the Web Search Assistant chatbot!
        I will try to answer your complex questions. 
        
        Enter x or q to quit at any point.
        """
    )
    load_dotenv()

    assistant_task = make_assistant_task(model, restart=False)
    search_task = make_search_task(model)
    critic_task = make_critic_task(model)

    def search_answer(qtool: QuestionTool) -> AnswerTool:
        """
        Take a QuestionTool, return an AnswerTool
        """
        return search_task[AnswerTool].run(qtool)

    def critic_feedback(fa: FinalAnswerTool) -> FeedbackTool:
        """
        Take a FinalAnswerTool, return a FeedbackTool
        """
        return critic_task[FeedbackTool].run(fa)

    def query_to_final_answer(question: str) -> FinalAnswerTool:
        """
        Take user's question, return FinalAnswerTool after
        iterating based on feedback from Critic.
        """

        question_tool_name = QuestionTool.default_value("request")
        final_answer_tool_name = FinalAnswerTool.default_value("request")

        tool = assistant_task[lr.ToolMessage].run(question)

        while True:
            if not isinstance(tool, (QuestionTool, FinalAnswerTool)):
                # no tool => nudge
                tool = assistant_task[lr.ToolMessage].run(
                    f"""
                     You forgot to use one of the tools:
                     `{question_tool_name}` or `{final_answer_tool_name}`.
                     """,
                )
            elif isinstance(tool, QuestionTool):
                # QuestionTool => get search result
                answer_tool = search_answer(tool)
                tool = assistant_task[lr.ToolMessage].run(answer_tool)
            else:
                # FinalAnswerTool => get feedback
                fb_tool = critic_feedback(tool)
                if fb_tool.suggested_fix == "":
                    # no suggested fix => return tool (which is a FinalAnswerTool)
                    return tool
                else:
                    # suggested fix => ask again
                    tool = assistant_task[lr.ToolMessage].run(fb_tool)

    # Interactive loop with user
    while True:
        question = Prompt.ask("What do you want to know?")
        if question.lower() in ["x", "q"]:
            break
        assistant_task.agent.init_state()
        final_answer = query_to_final_answer(question)
        assert isinstance(final_answer, FinalAnswerTool)


if __name__ == "__main__":
    app()
