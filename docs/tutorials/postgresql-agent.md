# PostgreSQL Agent using SQLChatAgent


The SQLChatAgent is designed to facilitate interactions with an SQL database using natural language. The agent uses the schema context to generate SQL queries based on a user's input. Here is a tutorial on how to set up an agent with your PostgreSQL database.

## Before you begin

!!! note "Data Privacy Considerations"
    Since the SQLChatAgent uses GPT-4 as the underlying language model, users should be aware that any database information processed by the agent may be sent to OpenAI's API and should therefore be comfortable with this data exchange.

Follow the [setup guide](/quick-start/setup/) to get started with Langroid.

## Initialize the agent

```python
from langroid.agent.special.sql.sql_chat_agent import (
    SQLChatAgent,
    SQLChatAgentConfig,
)

agent = SQLChatAgent(
            config=SQLChatAgentConfig(
                database_uri="postgresql://example.db",
            )
        )
```

## Configuration
!!! note "Strongly Reccomended"
* `context_descriptions`: Sets the schema context for the agent to use when generating queries  
```json
{
    "table1": {
        "description": "description of table1",
        "columns": {
            "column1": "description of column1 in table1",
            "column2": "description of column2 in table1"
        }
    },
    "employees": {
        "description": "The 'employees' table contains information about the employees. It relates to the 'departments' and 'sales' tables via foreign keys.",
        "columns": {
            "id": "A unique identifier for an employee. This ID is used as a foreign key in the 'sales' table.",
            "name": "The name of the employee.",
            "department_id": "The ID of the department the employee belongs to. This is a foreign key referencing the 'id' in the 'departments' table."
        }
    },
}
```
>By default, if no context description json file is provided in the config, the agent will automatically generate the file using the built-in Postgres table/column comments.

* `schema_tools`: Toggles a retrieval mode which enables the agent to systematically request for specific parts of the schema. When enabling this option, the agent performs the following steps:

    1. Asks for table names.
    2. Asks for table descriptions and column names from possibly relevant table names.
    3. Asks for column descriptions from possibly relevant columns.
    4. Writes the SQL query.

This approach is especially useful for large schemas that may have millions of tokens worth of descriptions, which is not feasible for large language models to handle. By selectively feeding only the relevant context descriptions, this mode optimizes token usage, though it may result in 2-3x more OpenAI API calls.


## Putting it all together  

In the code below, we will allow the agent to generate the context descriptions from table comments by excluding the `context_descriptions` config option.
```python
from langroid.agent.special.sql.sql_chat_agent import (
    SQLChatAgent,
    SQLChatAgentConfig,
)

# Initialize SQLChatAgent with a PostgreSQL database URI and enable schema_tools
agent = SQLChatAgent(gi
            config=SQLChatAgentConfig(
                database_uri="postgresql://example.db",
                schema_tools=True,
            )
        )

# Run the task to interact with the SQLChatAgent
task = Task(agent)
task.run()
```

By following this tutorial, you should now be able to set up an SQLChatAgent that interacts with a PostgreSQL database, making querying a seamless experience.