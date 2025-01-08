# Chat with a PostgreSQL DB using SQLChatAgent

The [`SQLChatAgent`](../reference/agent/special/sql/sql_chat_agent.md) is
designed to facilitate interactions with an SQL database using natural language.
A ready-to-use script based on the `SQLChatAgent` is available in the `langroid-examples` 
repo at [`examples/data-qa/sql-chat/sql_chat.py`](https://github.com/langroid/langroid-examples/blob/main/examples/data-qa/sql-chat/sql_chat.py)
(and also in a similar location in the main `langroid` repo).
This tutorial walks you through how you might use the `SQLChatAgent` if you were
to write your own script from scratch. We also show some of the internal workings of this Agent.

The agent uses the schema context to generate SQL queries based on a user's
input. Here is a tutorial on how to set up an agent with your PostgreSQL
database. The steps for other databases are similar. Since the agent implementation relies 
on SqlAlchemy, it should work with any SQL DB that supports SqlAlchemy.
It offers enhanced functionality for MySQL and PostgreSQL by 
automatically extracting schemas from the database. 

## Before you begin

!!! note "Data Privacy Considerations"
    Since the SQLChatAgent uses the OpenAI GPT-4 as the underlying language model,
    users should be aware that database information processed by the agent may be
    sent to OpenAI's API and should therefore be comfortable with this.
1. Install PostgreSQL dev libraries for your platform, e.g.
    - `sudo apt-get install libpq-dev` on Ubuntu,
    - `brew install postgresql` on Mac, etc.

2. Follow the general [setup guide](../quick-start/setup.md) to get started with Langroid
(mainly, install `langroid` into your virtual env, and set up suitable values in 
the `.env` file). Note that to use the SQLChatAgent with a PostgreSQL database,
you need to install the `langroid[postgres]` extra, e.g.:

    - `pip install "langroid[postgres]"` or 
    - `poetry add "langroid[postgres]"` or `uv add "langroid[postgres]"`
    - `poetry install -E postgres` or `uv pip install --extra postgres -r pyproject.toml`


If this gives you an error, try `pip install psycopg2-binary` in your virtualenv.


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

The following components of `SQLChatAgentConfig` are optional but strongly
recommended for improved results:

* `context_descriptions`: A nested dictionary that specifies the schema context for
  the agent to use when generating queries, for example:

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
  }
}
```

> By default, if no context description json file is provided in the config, the 
agent will automatically generate the file using the built-in Postgres table/column comments.

* `schema_tools`: When set to `True`, activates a retrieval mode where the agent
  systematically requests only the parts of the schemas relevant to the current query. 
  When this option is enabled, the agent performs the following steps:

    1. Asks for table names.
    2. Asks for table descriptions and column names from possibly relevant table
       names.
    3. Asks for column descriptions from possibly relevant columns.
    4. Writes the SQL query.

  Setting `schema_tools=True` is especially useful for large schemas where it is costly or impossible 
  to include the entire schema in a query context. 
  By selectively using only the relevant parts of the context descriptions, this mode
  reduces token usage, though it may result in 1-3 additional OpenAI API calls before
  the final SQL query is generated.

## Putting it all together

In the code below, we will allow the agent to generate the context descriptions
from table comments by excluding the `context_descriptions` config option.
We set `schema_tools` to `True` to enable the retrieval mode.

```python
from langroid.agent.special.sql.sql_chat_agent import (
    SQLChatAgent,
    SQLChatAgentConfig,
)

# Initialize SQLChatAgent with a PostgreSQL database URI and enable schema_tools
agent = SQLChatAgent(gi
config = SQLChatAgentConfig(
    database_uri="postgresql://example.db",
    schema_tools=True,
)
)

# Run the task to interact with the SQLChatAgent
task = Task(agent)
task.run()
```

By following these steps, you should now be able to set up an `SQLChatAgent`
that interacts with a PostgreSQL database, making querying a seamless
experience.

In the `langroid` repo we have provided a ready-to-use script
[`sql_chat.py`](https://github.com/langroid/langroid/blob/main/examples/data-qa/sql-chat/sql_chat.py)
based on the above, that you can use right away to interact with your PostgreSQL database:

```python
python3 examples/data-qa/sql-chat/sql_chat.py
```

This script will prompt you for the database URI, and then start the agent.

