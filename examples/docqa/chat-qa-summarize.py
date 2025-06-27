"""
Two-agent system to do Question-Answer based summarization of documents.
E.g. one could use this to summarize a very large document, assuming there is a
reasonable abstract/intro at the start that "covers" the import aspects.

WriterAgent (has no access to docs) is tasked with writing 5 bullet points based on
some docs. Initially it generates a summary of the docs from the beginning of the doc,
then it formulates questions to ask until it gets 5 key pieces of information.

DocAgent (has access to docs) answers these questions using RAG.

Run like this:

python examples/docqa/chat-qa-summarize.py

You can let it run and it will finish with 5 key bullet points about the document(s).

There are optional args, especially note you can pass in a different LLM model, e.g.

python3 examples/docqa/chat-qa-summarize.py -m ollama/nous-hermes2-mixtral

See here for how to set up a Local LLM to work with Langroid:
https://langroid.github.io/langroid/tutorials/local-llm-setup/

"""

import typer
from rich import print
import os

import langroid as lr
import langroid.language_models as lm
from langroid.parsing.urls import get_list_from_user
from langroid.utils.configuration import set_global, Settings

app = typer.Typer()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="specify alternative LLM, e.g. ollama/mistral",
    ),
) -> None:
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
    )
    config = lr.agent.special.DocChatAgentConfig(
        llm=llm_config,
        n_neighbor_chunks=2,
        n_similar_chunks=3,
        n_relevant_chunks=3,
        parsing=lr.parsing.parser.ParsingConfig(
            chunk_size=50,
            overlap=10,
            n_neighbor_ids=4,
        ),
    )
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )
    doc_agent = lr.agent.special.DocChatAgent(config)
    doc_agent.vecdb.set_collection("docqa-chat-multi", replace=True)
    print("[blue]Welcome to the document chatbot!")
    print("[cyan]Enter x or q to quit, or ? for evidence")
    print(
        """
        [blue]Enter some URLs or file/dir paths below (or leave empty for default URLs)
        """.strip()
    )
    inputs = get_list_from_user()
    if len(inputs) == 0:
        inputs = config.default_paths
    doc_agent.config.doc_paths = inputs
    doc_agent.ingest()
    topics_doc = doc_agent.summarize_docs(
        instruction="""
        Ignore the system message, and follow these instructions.
        Below is some text. Do not react to it. 
        Simply read it and give me a list of up to 3 main topics from the text,
        in the form of short NUMBERED SENTENCES.
        --------------------------------
        """,
    )
    topics = topics_doc.content
    doc_task = lr.Task(
        doc_agent,
        name="DocAgent",
        done_if_no_response=[lr.Entity.LLM],  # done if null response from LLM
        done_if_response=[lr.Entity.LLM],  # done if non-null response from LLM
        system_message="""You will receive various questions about some documents, and
        your job is to answer them concisely in at most 2 sentences, citing sources.
        """,
    )

    writer_agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm_config,
            vecdb=None,
        )
    )
    writer_task = lr.Task(
        writer_agent,
        # SET interactive to True to slow it down, but keep hitting enter to progress
        interactive=False,
        name="WriterAgent",
        system_message=f"""
        You have to collect some information from some documents, on these topics:
        {topics}
        However you do not have access to those documents, so you must ask me
        questions, ONE AT A TIME, and I will answer each question.
        Once you have collected 5 key pieces of information, say "DONE" and summarize 
        them in bullet points.  
        """,
    )

    validator_agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="Validator",
            llm=llm_config,
            system_message="""
            Your only task is to check whether the user's message consists of
            NO QUESTION, ONE question or MULTIPLE questions. This is how you must respond:
        
            - If the msg is NOT SEEKING any INFO, respond with this:
                "Please ask a SINGLE QUESTION about a topic you want to know about.
                Wait for the answer before asking your next question".
            - If user's msg contains just ONE question, or no question at all, say DONE
            - Otherwise (i.e there are MULTIPLE questions/requests for info),
              then respond with this:
            "Please ask only ONE question at a time. Ask your question again.
            Only when you have answers to all of your questions present your final
            bullet points saying  'DONE here are the bullet pts...'."
            
            IMPORTANT: DO NOT TRY TO ANSWER THE QUESTIONS YOURSELF.            
            """,
        ),
    )
    validator_task = lr.Task(validator_agent, interactive=False, single_round=True)

    writer_task.add_sub_task([validator_task, doc_task])

    writer_task.run()


if __name__ == "__main__":
    app()
