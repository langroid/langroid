# Specifying and Implementing LLM Agents

Created by: Prasad Chalasani
Created time: April 15, 2023 11:19 AM
Last edited by: Prasad Chalasani
Last edited time: April 16, 2023 8:28 AM

## Background

We wanted to build a simple app to query a set of documents using Langchain. Here are our observations from this experience:

- We had to hunt down multiple pages of documentation to find exactly what we needed to do
- We had to look at examples in multiple notebooks, to mix and match the functionality we were looking for
- The exact prompts used behind the scenes were deep inside the code
- Stepping through the debugger to see what is going on lead us down rabbit holes of complex class hierarchies.

***It should not have been this hard!***

This experience motivated us to develop a composable, hackable, flexible, transparent framework to build LLM-driven systems. The intended users are developers who like to see what is under the hood, and at the same time want a set of primitives they can work with at the right level of abstraction. 

As a first example of how this level of abstraction might work, we specify an implementation of this document-query system in the next section. In a later section we consider a text classifier.

---

## Example: Querying your docs

**Given:** document $D$
**Goal:** answer **standalone** query $q$ on document $D$

$$
a = \text{query}(D,q)
$$

where $a$ answers $q$ using info in $D$.

A framework like [Langchain](https://python.langchain.com/en/latest/index.html#) allows us to accomplish this in a few lines of code:

```python
llm = OpenAI(temperature=0)

text_splitter = CharacterTextSplitter(chunk_size=10000,	chunk_overlap=0)

texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                 chain_type="stuff",
                                 retriever=vectorstore.as_retriever())

qa.run("What is the total cost of tuition at Rutgers?")
```

However, these statements lack transparency regarding the specifics of how it is achieved. For example, the prompts being created to accomplish this are not visible. To modify them, one must refer to the Langchain documentation to explore other types of "chains" that are available and then locate and modify the code to fit our requirements.

A more transparent approach is to expose the different primitives, building blocks, and prompts so that developers can customize them according to their needs. One such specification is shown below. This can be viewed as high-level Python code, which could itself be the output of processing a higher-level (declarative?) specification.

1. Preprocess: store embeddings of $k$ chunks of $D$ in a vector database
    
    ```python
    d[1:k] = split(D, k) # k roughly equal size chunks
    For each i in [k]:
    	e[i] = embed(d[i])
    	vecDB.insert(e[i], d[i]) # attach d[i] as metadata to e[i]
    ```
    
2. Respond to query $q$
a. Find relevant doc-chunks from vecDB:
    
    ```
     e = embed(q)
     e[1:m] = vecDB.nearest(e,m) # find m nearest neighbor vectors
     d[1:m] = vecDB.lookup(d[1:m]) # find the associated chunks
    
    ```
    
    b. Extract relevant verbatim text from each chunk d[i] with LLM
    
    ```
     For each i in [m]:
         x[i] = LLM.extract(d[i], q)
    ```
    
    c. Compose final answer
    
    ```python
    a = LLM.compose(x[1:m], q)
    ```
    

In the next sections we define the primitives `LLM.extrac()` and `LLM.compose()`

### Define `LLM.extract`

`LLM.extract(d,q):` extract verbatim text relevant to query q from document (or chunk) d

We implement this via the “core” primitive of any LLM, namely text-completion, which we denote as `complete` , using a special `ExtractPrompt`:

```
ExtractPrompt[d,q] =
    "Here is a part of a long doc,
    see if it contains text relevant to the question. 
		Return any relevant text verbatim.
    Content: {d}
    Question: {q}

    Relevant text, if any:"

LLM.extract(d,q) = LLM.complete(ExtractPrompt[d,q])
```

### Define `LLM.compose`

`LLM.compose(d[1:m], q):` compose an answer to q based on text fragments d[1:m]

We implement this via the `complete` primitive of the LLM, with a special "ComposePrompt", which includes some few-shot examples.

```
ComposePrompt[d[1:m],q] =
    "Here are some portions of a long document, relevant to a question.
    Return the final answer. Some examples are below.
    Content: …
    Content: …
    Question: when was Washington born?
    Final Answer: Washington was born in 1750
    Content: …
    Content: …
    Question: What did Biden say about Obama?
    Final Answer: Bid did not mention Obama
    Content: {d[1]}
    Content: {d[2]}
    …
    Content: {d[m]}
    Question: {q}
    Final answer:"

LLM.compose(d[1:m],q) := LLM.complete(ComposePrompt[d[1:m],q))`
```

### Follow-up questions

There are two ways to handle follow-up questions in a dialog with an LLM

- Pass in the entire dialog history, along with any contextual documents, to the LLM as context, and add the current query as the “actual” question needing a response
- First convert the follow-up question to a stand-alone question, and then use the standalone-query approach in the first section to answer the question.

The first approach can be expensive in terms of token cost. We consider the second approach.

**************Given:************** 

- A Dialog, i.e. a sequence of question-response pairs $D = \{(q_i, a_i)\}$ where $q_i$ is the human query, and $a_i$ is the LLM response
- a follow-up question $f$

**********************Goal:********************** convert $f$ to a ************stand-alone************ question $q$ 

For this we define an LLM primitive `LLM.standAlone(D, f)`which can be implemented as follows.

```python
DialogToStandAlonePrompt[D,f] = 
	"Given the following conversation and a follow-up question,
  rephrase the follow-up question to be a stand-alone question:
	Human: {q_1}
	Assistant: {a_1}
	Human: {q_2}
	Assistant: {a_2}
  ...
	Human: {q_m}
	Assistant: {a_m}
  Follow-up Input: <f>
  Standalone question:"

LLM.standAlone(D,f) = LLM.complete(DialogToStandAlonePrompt[D,f]) 	

```

---

## A text classifier

Given a text passage, label it as one of $m$ categories.

Here is how we could leverage the LM + Retrieval Model (vectorDB) to do text classification.

Over time, collect human-labeled examples of (doc, category) pairs $(d_i, c_i)$ where $c_i \in [m]$. As we see each such pair, we store the embedding-vector of the doc $d_i$ into our vecDB, with metadata $c_i$ attached:

```python
Preprocessing: for each (doc, category) labeled pair (d_i, c_i):
	e_i = embed(d_i)
	vecDB.insert(e_i, c_i) # attach metadata c_i to e_i
```

When we see a new document $d$ , we categorize it using the LLM as follows. 

1. Map $d$ to an embedding vector and for each category $i \in [m]$ , find the closest $k$ documents in category $i$ in the vecDB.

```python
e = embed(d)
for each i in 1..m: 
		#find k nearest neighbor vectors of e in category i
		N(e,i)[1:k] = vecDB.nearest(e, k, i)
		D(e,i)[1:k] = vecDB.lookup( N(e,i)[1:k] ) # look up corresponding docs

	

```

For simplicity, refer to the docs D(e,i)[1:k]  above as a collection of labeled docs (d_i, c_i) where i ranges from 1 to n

1. Construct a prompt containing few-shot classification examples from these nearest neighbors, and ask the LLM to classify the document $d$. 
    
    ```python
    ClassificationPrompt[d[1:n], c[1:n], d] =
    		"Here are examples of context and category. 
        Label the last content with the right category.
    		content: {d_1}
    		category: {c_1}
    		content: {d_2}
    		category: {c_2}
    		...
    		content: {d_n}
    		category: {c_n}
    		content: {d}
    		category: "
    
    category = LLM.complete(ClassificationPrompt[d[1:n], c[1:n], d] )
    		
    ```
    

Note, we could have just done this with a “frozen” set of few-shot classification examples in the prompt, but here we are constructing few-shot examples on the fly on *neighbors* of the current query. The intuition is that this might help the LLM do a better job with the classification, due to two effects:

- the examples are similar to the given query,
- the few-shot examples are in a sense close to the decision-boundaries (i.e. potentially confusing), so by giving these, the LLM is forced to better discriminate among the categories.

We haven’t yet implemented this, but here is an actual interaction with GPT4, for classification with few-shot examples, where we classify sentences as “interesting” or not.

- Prompt:
    
    Below are some examples of whether or not certain passages are "interesting" or "not".  Based on these, classify the final passage as interesting or not.
    Passage: Language models are the hottest topic.
    Interesting? Yes
    Passage: The elections are tomorrow.
    interesting? No
    Passage: The new york giants are playing tomorrow:
    Interesting? No
    Passage:  ChatGPT outperforms middle-school kids on writing tasks. What does it mean for education?
    Interesting? Yes
    Passage: There is an existential threat to humanity arising from the prospect of being able to run quantum simulation on a quantum computers in the future.
    Interesting? Yes
    Passage: Here is a FREE opportunity to learn how to use GPT-4 and create AI applications:
    Interesting? Yes
    Passage: All of my friends who don’t work in tech are actively using ChatGPT or other AI powered tools for their jobs.
    Interesting?
    
- Response: 
Yes
- Prompt:
why?
- Response:
The passage "All of my friends who don’t work in tech are actively using ChatGPT or other AI-powered tools for their jobs" is classified as interesting because it suggests a widespread adoption and impact of AI-powered tools, including ChatGPT, across various industries and professions outside of the tech sector. This widespread adoption and the impact on non-tech industries is a topic that is likely to be of interest to readers, as it speaks to the broader influence of AI on society and the workforce. Additionally, the passage raises questions about how AI is changing the way people work and how it is being integrated into various job functions, making it an interesting topic for discussion and exploration.
- Prompt: But is it interesting according to the examples I gave?
- Response:
    
    Yes, the passage is interesting according to the examples provided. In the examples given, passages that are classified as interesting tend to involve topics related to technology and AI, their impact on society or specific industries, and their potential implications. For instance, the passages about language models being a hot topic, ChatGPT outperforming middle-school kids, the existential threat of quantum simulation, and the opportunity to learn GPT-4 are all considered interesting.
    
    The passage "All of my friends who don’t work in tech are actively using ChatGPT or other AI-powered tools for their jobs" fits within the same theme of discussing the impact and significance of AI and technology. It touches on the idea that AI-powered tools, including ChatGPT, have become so pervasive that they are now being used by individuals in industries outside of technology. This suggests a broader societal impact of AI, which aligns with the characteristics of the passages that were labeled as interesting in the examples provided.
    

### Remarks:

1. It’s nice to see that we can even get an ************explanation************ of the classification
2. The one problem with the above scheme is that we are passing entire documents into the LLM. This will be expensive or infeasible when documents are large. The approach should work however for short documents, or document-shards, or tweets, or short messages. Maybe for storing a (doc, category) pair for larger doc, we could extract part of that doc somehow (need to think about this).