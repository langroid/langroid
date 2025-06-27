"""
Demonstrating the utility of Hypothetical Questions (HQ) in the context of a
DocChatAgent.

In the following example, a DocChatAgent is created and it can be queried on its
documents both in a normal way and in a hypothetical way.

Although this is being referred to as Hypothetical Questions, it is not limited to
just questions -- it is simply a way to augment the document-chunks at ingestion time,
with keywords that increase the "semantic surface" of the chunks to improve
retrieval accuracy.

This example illustrates the benefit of HQ in a medical scenario
where each "document chunk" is simply the name of a medical test
(e.g. "cholesterol", "BUN", "PSA", etc)
and when `use_hypothetical_question` is enabled,
the chunk (i.e. test name) is augment it with keywords that add more
context, such as which organ it is related to
(e.g., "heart", "kidney", "prostate", etc).
This way, when a user asks "which tests are related to kidney health",
these augmentations ensure that the test names are retrieved more accurately.

Running the script compares the accuracy of
results of the DocChatAgent with and without HQ.

Run like this to use HQ:

python3 examples/docqa/hypothetical_questions.py

or without HQ:

python3 examples/docqa/hypothetical_questions.py --no-use-hq
"""

import typer
from rich import print
from rich.table import Table

import langroid as lr
import langroid.language_models as lm
from langroid.agent.batch import run_batch_function
from langroid.agent.special.doc_chat_agent import (
    ChunkEnrichmentAgentConfig,
    DocChatAgent,
    DocChatAgentConfig,
)
from langroid.parsing.parser import ParsingConfig
from langroid.utils.configuration import Settings
from langroid.vector_store.qdrantdb import QdrantDBConfig

app = typer.Typer()

lr.utils.logging.setup_colored_logging()

ORGAN = "kidney"


def setup_vecdb(docker: bool, reset: bool, collection: str) -> QdrantDBConfig:
    """Configure vector database."""
    return QdrantDBConfig(
        collection_name=collection, replace_collection=reset, docker=docker
    )


def run_document_chatbot(
    model: str,
    docker: bool,
    reset: bool,
    collection: str,
    use_hq: bool,
) -> None:
    """
    Main function for the document chatbot.

    Args:
        model: chat model
        docker: use docker for vector database
        reset: reset conversation memory
        collection: collection name
        use_hq: use hypothetical
    """
    llm_config = lm.OpenAIGPTConfig(chat_model=model)
    vecdb_config = setup_vecdb(docker=docker, reset=reset, collection=collection)
    enrichment_config = ChunkEnrichmentAgentConfig(
        batch_size=10,
        system_message="""
        You are an experienced clinical physician, very well-versed in
        medical tests and their names.
        You will be asked to identify WHICH ORGAN(s) Function/Health
        a test name is most closely associated with, to aid in 
        retrieving the medical test names more accurately from an embeddings db
        that contains thousands of such test names.
        The idea is to use the ORGAN NAME(S) provided by you, 
        to make the right test names easier to discover via keyword-matching
        or semantic (embedding) similarity.
         Your job is to generate up to 3 ORGAN NAMES
         MOST CLOSELY associated with the test name shown, ONE PER LINE.
         DO NOT SAY ANYTHING ELSE, and DO NOT BE OBLIGATED to provide 3 organs --
         if there is just one or two that are most relevant, that is fine.
        Examples:
          "cholesterol" -> "heart function", 
          "LDL" -> "artery health", etc,
          "PSA" -> "prostate health", 
          "TSH" -> "thyroid function", etc.                
        """,
        enrichment_prompt_fn=lambda test: f"""
        Which ORGAN(S) Function/Health is the medical test named 
        '{test}' most closely associated with?
        """,
    )

    config = DocChatAgentConfig(
        llm=llm_config,
        vecdb=vecdb_config,
        hypothetical_answer=False,
        rerank_diversity=False,
        rerank_periphery=False,
        use_reciprocal_rank_fusion=False,
        n_similar_chunks=10,
        n_relevant_chunks=10,
        parsing=ParsingConfig(
            chunk_size=120,
            overlap=15,
            min_chunk_chars=50,
        ),
        # n_neighbor_chunks=1,
        chunk_enrichment_config=enrichment_config if use_hq else None,
        relevance_extractor_config=None,
    )

    doc_agent = DocChatAgent(config=config)
    medical_tests = """
    BUN, Creatinine, GFR, ALT, AST, ALP, Albumin, Bilirubin, CBC, eGFR, PTH, 
    Uric Acid, Ammonia, Protein/Creatinine Ratio, Total Protein, LDH, SPEP, CRP, 
    ESR, Cystatin C
    """

    medical_test_list = [test.strip() for test in medical_tests.split(",")]

    # already "chunked" docs:
    docs = [lr.Document.from_string(test, is_chunk=True) for test in medical_test_list]
    # this should augment each test name with organ names that help improve retrieval
    doc_agent.ingest_docs(docs)
    if use_hq:
        print("[cyan]Test names augmented with organ names:")
        for doc in doc_agent.chunked_docs:
            print(doc.content)
            print("---")

    user_query = f"Which tests are related to {ORGAN} function?"

    _, relevant_chunks = doc_agent.get_relevant_extracts(user_query)
    relevant_chunks_str = "\n".join([chunk.content for chunk in relevant_chunks])
    print(f"relevant test names retrieved:\n{relevant_chunks_str}")
    system_msg = f"""
      You are an experienced clinical physician, well-versed in
      medical tests and their names. You are looking a set of 
      tests or readings that have been performed on a patient. 
      Based on these tests or readings, you need to determine 
      which of the tests shown are relevant to compiling a medical 
      report on the {ORGAN} function and {ORGAN} health of the 
      patient.
    """
    asst_msg = f"""
    Yes I perfectly understand! I will be diligent and discriminating, 
    and will accurately pick out which of the tests are related to
    compiling a comprehensive medical report on the {ORGAN} function and 
    {ORGAN} health. Please show me the full list of tests and/or readings
    and I PROMISE I will be able to tell you which of them are relevant to
    {ORGAN} function or {ORGAN} health.
    """
    user_msg = f"""
    Your patient had a series of tests/measurements performed,
    and below are the TEST (or measurement) NAMES that were recorded.
    For you to compile a comprehensive medical report on the {ORGAN} function and
    {ORGAN} health of the patient,
    which of these tests are typically considered related to this organ's 
    function or health? 
    
    Simply list the relevant test-names, VERBATIM exactly as they appear,
    one per line, without any explanation or elaboration.

    TESTS/MEASUREMNTS:

    {relevant_chunks_str}
    """

    retrieval_answer = doc_agent.llm.chat(
        [
            lm.LLMMessage(content=system_msg, role=lm.Role.SYSTEM),
            lm.LLMMessage(content=asst_msg, role=lm.Role.ASSISTANT),
            lm.LLMMessage(content=user_msg, role=lm.Role.USER),
        ]
    ).message
    print(f"\n\nAnswer from DocChatAgent.llm after retrieval:\n{retrieval_answer}")
    retrieval_tests = retrieval_answer.split("\n")
    retrieval_tests = [
        test.strip()
        for test in retrieval_tests
        if test.strip() and test.strip() in medical_test_list
    ]

    # compare this with directly asking the LLM about each individual test
    print(f"[blue]Directly asking the LLM whether each test is related to {ORGAN}:")
    llm = doc_agent.llm

    def llm_classify(test: str) -> str:
        return llm.chat(
            [
                lm.LLMMessage(content=system_msg, role=lm.Role.SYSTEM),
                lm.LLMMessage(content=asst_msg, role=lm.Role.ASSISTANT),
                lm.LLMMessage(
                    content=f"""
                          Is the medical test named '{test}' typically considered
                          DIRECTLY related to {ORGAN} function?,
                          simply say 'yes' or 'no'
                          """,
                    role=lm.Role.USER,
                ),
            ]
        ).message

    classifications = run_batch_function(llm_classify, medical_test_list, batch_size=5)
    direct_llm_tests = [
        test
        for test, classification in zip(medical_test_list, classifications)
        if "yes" in classification.lower()
    ]
    print("[green]Relevant tests from direct LLM query:\n")
    print("\n".join(direct_llm_tests))

    # Create a table with test comparison
    test_union = set(direct_llm_tests).union(set(retrieval_tests))

    with_str = "with" if use_hq else "without"
    table = Table(
        title=f"Test Detection Methods Comparison for {ORGAN} {with_str} Hyp Questions"
    )
    table.add_column("Test", justify="left")
    table.add_column("Direct", justify="center")
    table.add_column("Retrieval", justify="center")

    for test in sorted(test_union):
        direct = "x" if test in direct_llm_tests else ""
        retrieved = "x" if test in retrieval_tests else ""
        table.add_row(test, direct, retrieved)

    print("\n")
    print(table)

    # calc percent overlap or jacard similarity between the two sets of relevant tests
    overlap = len(
        set(direct_llm_tests).intersection(set(relevant_chunks_str.split("\n")))
    )
    union = len(test_union)
    jacard_pct = (100 * overlap / union) if union > 0 else 0
    print(
        f"[cyan]Jaccard similarity between the two sets of relevant tests: {jacard_pct:.2f}%"
    )


@app.command()
def main(
    debug: bool = typer.Option(
        False, "--debug/--no-debug", "-d", help="Enable debug mode"
    ),
    stream: bool = typer.Option(
        True, "--stream/--no-stream", "-s", help="Enable streaming output"
    ),
    cache: bool = typer.Option(True, "--cache/--no-cache", "-c", help="Enable caching"),
    model: str = typer.Option(
        lm.OpenAIChatModel.GPT4o_MINI.value, "--model", "-m", help="Chat model to use"
    ),
    collection: str = typer.Option(
        "docchat_hq", "--collection", help="Collection name for vector database"
    ),
    docker: bool = typer.Option(
        True, "--docker/--no-docker", help="Use docker for vector database"
    ),
    reset: bool = typer.Option(
        True, "--reset/--no-reset", help="Reset conversation memory"
    ),
    use_hq: bool = typer.Option(
        True, "--use-hq/--no-use-hq", help="Use hypothetical questions"
    ),
) -> None:
    """Main app function."""
    lr.utils.configuration.set_global(
        Settings(
            debug=debug,
            cache=cache,
            stream=stream,
        )
    )

    run_document_chatbot(
        model=model,
        docker=docker,
        collection=collection,
        reset=reset,
        use_hq=use_hq,
    )


if __name__ == "__main__":
    app()
