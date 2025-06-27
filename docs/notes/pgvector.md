
---

## **Setup PostgreSQL with pgvector using Docker**

To quickly get a PostgreSQL instance with pgvector running, the easiest method is to use Docker. Follow the steps below:

### **1. Run PostgreSQL with Docker**

Use the official `ankane/pgvector` Docker image to set up PostgreSQL with the pgvector extension. Run the following command:

```bash
docker run --name pgvector -e POSTGRES_USER=your_postgres_user -e POSTGRES_PASSWORD=your_postgres_password -e POSTGRES_DB=your_database_name -p 5432:5432 ankane/pgvector
```

This will pull the `ankane/pgvector` image and run it as a PostgreSQL container on your local machine. The database will be accessible at `localhost:5432`. 

### **2. Include `.env` file with PostgreSQL credentials**

These environment variables should be same which were set while spinning up docker container.
Add the following environment variables to a `.env` file for configuring your PostgreSQL connection:

```dotenv
POSTGRES_USER=your_postgres_user
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_DB=your_database_name
```
## **If you want to use cloud offerings of postgres**

We are using **Tembo** for demonstrative purposes here.  

### **Steps to Set Up Tembo**  
Follow this [quickstart guide](https://tembo.io/docs/getting-started/getting_started) to get your Tembo credentials.  

1. Sign up at [Tembo.io](https://cloud.tembo.io/).  
2. While selecting a stack, choose **VectorDB** as your option.  
3. Click on **Deploy Free**.  
4. Wait until your database is fully provisioned.  
5. Click on **Show Connection String** to get your connection string.  

### **If you have connection string, no need to setup the docker**
Make sure your connnection string starts with `postgres://` or `postgresql://`

Add this to your `.env`
```dotenv
POSTGRES_CONNECTION_STRING=your-connection-string
```

---

## **Installation**

If you are using `uv` or `pip` for package management, install Langroid with postgres extra:

```bash
uv add langroid[postgres]  # or
pip install langroid[postgres]
```

---

## **Code Example**

Here's an example of how to use Langroid with PostgreSQL:

```python
import langroid as lr
from langroid.agent.special import DocChatAgent, DocChatAgentConfig
from langroid.embedding_models import OpenAIEmbeddingsConfig

# Configure OpenAI embeddings
embed_cfg = OpenAIEmbeddingsConfig(
    model_type="openai",
)

# Configure the DocChatAgent with PostgresDB
config = DocChatAgentConfig(
    llm=lr.language_models.OpenAIGPTConfig(
        chat_model=lr.language_models.OpenAIChatModel.GPT4o
    ),
    vecdb=lr.vector_store.PostgresDBConfig(
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

## **Create and Ingest Documents**

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

## **Get an Answer from the LLM**

Now that documents are ingested, you can query the agent to get an answer.

### **Code Example**

```python
answer = agent.llm_response("When will the new ice age begin?")
```

---
