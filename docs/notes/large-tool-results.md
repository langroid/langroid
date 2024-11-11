# Handling large tool results

In some cases, the result of handling a `ToolMessage` could be very large,
e.g. when the Tool is a database query that returns a large number of rows,
or a large schema. When used in a task loop, this large result may then be
sent to the LLM to generate a response.