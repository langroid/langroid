# Enriching Chunked Documents for Better Retrieval

Available in Langroid v0.34.0 or later. 

When using the `DocChatAgent` for RAG with documents in highly specialized/technical
domains, retrieval accuracy may be low since embeddings are not sufficient to capture 
relationships between entities, e.g. suppose a document-chunk consists of a medical 
test name "BUN" (Blood Urea Nitrogen), and a retrieval query is looking for 
tests related to kidney function, the embedding for "BUN" may not be close to the
embedding for "kidney function", and the chunk may not be retrieved.

In such cases it is useful to *enrich* the chunked documents with additional keywords
(or even "hypothetical questions") to increase the "semantic surface area" of the chunk,
so that the chunk is more likely to be retrieved for relevant queries.

As of Langroid v0.34.0, you can provide a `chunk_enrichment_config` 
of type `ChunkEnrichmentAgentConfig`, in the `DocChatAgentConfig`. 
This config extends `ChatAgentConfig` and has the following fields:

- `batch_size` (int): The batch size for the chunk enrichment agent. Default is 50.
- `delimiter` (str): The delimiter to use when 
   concatenating the chunk and the enriched text. 
- `enrichment_prompt_fn`: function (`str->str`) that creates a prompt
  from a doc-chunk string `x`

In the above medical test example, suppose we want to augment a chunk containing
only the medical test name, with the organ system it is related to. We can set up
a `ChunkEnrichmentAgentConfig` as follows:

```python
from langroid.agent.special.doc.doc_chat_agent import (
    ChunkEnrichmentAgentConfig,
)

enrichment_config = ChunkEnrichmentAgentConfig(
    batch_size=10,
    system_message=f"""
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

doc_agent_config = DocChatAgentConfig(
    chunk_enrichment_config=enrichment_config,
    ...
)
```

This works as follows:

- Before ingesting document-chunks into the vector-db, a specialized 
  "chunk enrichment" agent is created, configured with the `enrichment_config` above.
- For each document-chunk `x`, the agent's `llm_response_forget_async` method is called
 using the prompt created by `enrichment_prompt_fn(x)`. The resulting response text 
 `y` is concatenated with the original chunk text `x` using the `delimiter`,
  before storing in the vector-db. This is done in batches of size `batch_size`.
- At query time, after chunk retrieval, before generating the final LLM response,
  the enrichments are stripped from the retrieved chunks, and the original content
  of the retrieved chunks are passed to the LLM for generating the final response.

See the script 
[`examples/docqa/doc-chunk-enrich.py`](https://github.com/langroid/langroid/blob/main/examples/docqa/doc-chunk-enrich.py)
for a complete example. Also see the tests related to "enrichment" in 
[`test_doc_chat_agent.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_doc_chat_agent.py).

