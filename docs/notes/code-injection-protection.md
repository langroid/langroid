# Code Injection Protection with full_eval Flag

Available in Langroid since v0.53.15.

Langroid provides a security feature that helps protect against code injection vulnerabilities when evaluating pandas expressions in `TableChatAgent` and `VectorStore`. This protection is controlled by the `full_eval` flag, which defaults to `False` for maximum security, but can be set to `True` when working in trusted environments.

## Background

When executing dynamic pandas expressions within `TableChatAgent` and in `VectorStore.compute_from_docs()`, there is a risk of code injection if malicious input is provided. To mitigate this risk, Langroid implements a command sanitization system that validates and restricts the operations that can be performed.

## How It Works

The sanitization system uses AST (Abstract Syntax Tree) analysis to enforce a security policy that:

1. Restricts DataFrame methods to a safe whitelist
2. Prevents access to potentially dangerous methods and arguments
3. Limits expression depth and method chaining
4. Validates literals and numeric values to be within safe bounds
5. Blocks access to any variables other than the provided DataFrame

When `full_eval=False` (the default), all expressions are run through this sanitization process before evaluation. When `full_eval=True`, the sanitization is bypassed, allowing full access to pandas functionality.

## Configuration Options

### In TableChatAgent

```python
from langroid.agent.special.table_chat_agent import TableChatAgentConfig, TableChatAgent

config = TableChatAgentConfig(
    data=my_dataframe,
    full_eval=False,  # Default: True only for trusted input
)

agent = TableChatAgent(config)
```

### In VectorStore

```python
from langroid.vector_store.lancedb import LanceDBConfig, LanceDB

config = LanceDBConfig(
    collection_name="my_collection",
    full_eval=False,  # Default: True only for trusted input
)

vectorstore = LanceDB(config)
```

## When to Use full_eval=True

Set `full_eval=True` only when:

1. All input comes from trusted sources (not from users or external systems)
2. You need full pandas functionality that goes beyond the whitelisted methods
3. You're working in a controlled development or testing environment

## Security Considerations

- By default, `full_eval=False` provides a good balance of security and functionality
- The whitelisted operations support most common pandas operations
- Setting `full_eval=True` removes all protection and should be used with caution
- Even with protection, always validate input when possible

## Affected Classes

The `full_eval` flag affects the following components:

1. `TableChatAgentConfig` and `TableChatAgent` - Controls sanitization in the `pandas_eval` method
2. `VectorStoreConfig` and `VectorStore` - Controls sanitization in the `compute_from_docs` method
3. All implementations of `VectorStore` (ChromaDB, LanceDB, MeiliSearch, PineconeDB, PostgresDB, QdrantDB, WeaviateDB)

## Example: Safe Pandas Operations

When `full_eval=False`, the following operations are allowed:

```python
# Allowed operations (non-exhaustive list)
df.head()
df.groupby('column')['value'].mean()
df[df['column'] > 10]
df.sort_values('column', ascending=False)
df.pivot_table(...)
```

Some operations that might be blocked include:

```python
# Potentially blocked operations
df.eval("dangerous_expression")
df.query("dangerous_query")
df.apply(lambda x: dangerous_function(x))
```

## Testing Considerations

When writing tests that use `TableChatAgent` or `VectorStore.compute_from_docs()` with pandas expressions that go beyond the whitelisted operations, you may need to set `full_eval=True` to ensure the tests pass.