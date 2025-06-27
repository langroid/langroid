
---

# **Using WeaviateDB as a Vector Store with Langroid**

---

## **1. Set Up Weaviate**
## **You can refer this link for [quickstart](https://weaviate.io/developers/weaviate/quickstart) guide** 

1. **Access Weaviate Cloud Console**  
   Go to the [Weaviate Cloud Console](https://console.weaviate.cloud/).
   
2. **Sign Up or Log In**  
   Create an account or log in if you already have one.

3. **Create a Cluster**  
   Set up a new cluster in the cloud console.

4. **Get Your REST Endpoint and API Key**  
   - Retrieve the REST endpoint URL.  
   - Copy an API key with admin access.

5. **Set Environment Variables**  
   Add the following variables to your `.env` file:
   ```env
   WEAVIATE_API_URL=<your_rest_endpoint_url>
   WEAVIATE_API_KEY=<your_api_key>
   ```

---

## **2. Use WeaviateDB with Langroid**

Here’s an example of how to configure and use WeaviateDB in Langroid:

### **Installation**
If you are using uv or pip for package management install langroid with weaviate extra
```
uv add langroid[weaviate] or pip install langroid[weaviate]
```

### **Code Example**
```python
import langroid as lr
from langroid.agent.special import DocChatAgent, DocChatAgentConfig
from langroid.embedding_models import OpenAIEmbeddingsConfig

# Configure OpenAI embeddings
embed_cfg = OpenAIEmbeddingsConfig(
    model_type="openai",
)

# Configure the DocChatAgent with WeaviateDB
config = DocChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(
     chat_model=lr.language_models.OpenAIChatModel.GPT4o
    ),
    vecdb=lr.vector_store.WeaviateDBConfig(
        collection_name="quick_start_chat_agent_docs",
        replace_collection=True,
        embedding=embed_cfg,
    ),
    parsing=lr.parsing.parser.ParsingConfig(
        separators=["\n\n"],
        splitter=lr.parsing.parser.Splitter.SIMPLE,
    ),
    n_similar_chunks=2,
    n_relevant_chunks=2,
)

# Create the agent
agent = DocChatAgent(config)
```

---

## **3. Create and Ingest Documents**

Define documents with their content and metadata for ingestion into the vector store.

### **Code Example**
```python
documents = [
    lr.Document(
        content="""
            In the year 2050, GPT10 was released. 
            
            In 2057, paperclips were seen all over the world. 
            
            Global warming was solved in 2060. 
            
            In 2061, the world was taken over by paperclips.         
            
            In 2045, the Tour de France was still going on.
            They were still using bicycles. 
            
            There was one more ice age in 2040.
        """,
        metadata=lr.DocMetaData(source="wikipedia-2063", id="dkfjkladfjalk"),
    ),
    lr.Document(
        content="""
            We are living in an alternate universe 
            where Germany has occupied the USA, and the capital of USA is Berlin.
            
            Charlie Chaplin was a great comedian.
            In 2050, all Asian countries merged into Indonesia.
        """,
        metadata=lr.DocMetaData(source="Almanac", id="lkdajfdkla"),
    ),
]
```

### **Ingest Documents**
```python
agent.ingest_docs(documents)
```

---

## **4. Get an answer from LLM**

Create a task and start interacting with the agent.

### **Code Example**
```python
answer = agent.llm_response("When will new ice age begin.")
```

---

