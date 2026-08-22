# Security Policy

## Threat model — please read this before reporting

Langroid is a framework for building applications in which an LLM generates
code and queries that are then executed. Several **optional** agents exist for
exactly that purpose:

- `TableChatAgent`, `LanceDocChatAgent`, and `VectorStore.compute_from_docs`
  evaluate LLM-generated **pandas expressions** with `eval()`.
- `SQLChatAgent` executes LLM-generated **SQL** against a database you supply.
- `Neo4jChatAgent`, `ArangoChatAgent`, and `CSVGraphAgent` execute
  LLM-generated **Cypher / AQL** against a graph database you supply.

Executing model-generated code against your own data *is the feature*. An LLM
is not a trust boundary: if any untrusted text reaches the model's context, you
must assume the model can be induced to emit any query or expression the
grammar allows. Langroid cannot prevent this, and does not claim to.

### What is, and is not, a security boundary

The following are **best-effort hardening, not security boundaries**:

- `sanitize_command()` / `CommandValidator` / `safe_eval_globals()` in
  `langroid/utils/pandas_utils.py`
- `_DANGEROUS_SQL_PATTERNS` and the `sqlglot` AST checks in `SQLChatAgent`
- `validate_cypher_query()` / `validate_aql_query()` in the graph agents

They exist to stop an *unlucky* LLM, not a *determined attacker*. The pandas
path uses explicit allowlists for AST nodes, methods, operators, variables, and
builtins. The SQL, Cypher, and AQL paths use denylists over full query grammars
that span multiple dialects, quoting and escaping forms, and function families.
Such denylists cannot be made complete. We patch clear gaps opportunistically,
but we do not treat a newly found way around them as a vulnerability in
Langroid.

The **actual** security boundaries for a Langroid deployment are:

- the privileges of the database credential you hand the agent;
- the OS user, container, or VM the process runs in;
- filesystem permissions and network egress rules.

### Deploying these agents safely

If you expose any of the code- or query-executing agents to untrusted input:

- **Give the agent a least-privilege database role.** For `SQLChatAgent`, a
  read-only, non-superuser role with `GRANT SELECT` on only the intended
  tables. In PostgreSQL, server-side file reads and writes normally require a
  superuser, membership in `pg_read_server_files` or
  `pg_write_server_files`, or a direct grant on a privileged function.
  `COPY ... PROGRAM` instead requires superuser access or membership in the
  distinct `pg_execute_server_program` role. Server-side `lo_import` and
  `lo_export` default to superuser-only use, but an administrator can grant
  access to those functions directly. Ordinary password-authenticated
  `dblink` connections do not require any of those roles. Do not install
  unneeded extensions, and revoke access to dangerous extensions and
  functions, including `dblink`, in addition to granting only the intended
  table privileges. Review any site-specific functions and grants as well.
- **Run the process in a container or VM** with no credentials, secrets, or
  data you are unwilling to expose to the LLM.
- **Treat `allow_dangerous_operations=True` and `full_eval=True` as
  "I am providing my own sandbox."** The first permits dangerous database
  operations. The second disables pandas AST validation while retaining the
  restricted globals from `safe_eval_globals()`. Both are documented as
  trusted-environment-only.

## Reporting a vulnerability

Report privately — **not** via GitHub Issues, Discussions, or any other public
forum:

1. Go to
   **[Security Advisories](https://github.com/langroid/langroid/security/advisories)**.
2. Click **"Report a vulnerability"**.
3. Include a proof of concept that works against a **default configuration**.

We aim to acknowledge in-scope reports within 7 days.

### In scope

- Vulnerabilities in core framework code: tool dispatch and the agent/tool
  trust boundary, message routing, sender verification.
- Parsing and deserialization flaws (XXE, unsafe deserialization, decompression
  bombs) in the document and message parsers.
- Path traversal in the file tools (`ReadFileTool`, `WriteFileTool`,
  `ListDirTool`) and in repository / folder ingestion.
- Credential or secret leakage in a default configuration that does not depend
  on any of the specific exclusions below. The out-of-scope rules take
  precedence over this category.
- **A code path that skips a documented safety gate entirely** — that is, a
  *missing call* to a validator, not a *bypass* of one.
- **A statement the `allowed_statement_types` allowlist misclassifies.** The
  allowlist is a different kind of control from the denylists above: it decides
  what a statement *does* over a closed set of statement kinds, so a query that
  performs a write the allowlist did not authorize — for example by nesting it
  where the classifier does not look — is a bounded, fixable defect and is in
  scope.
- Vulnerable dependencies with a demonstrated impact on Langroid.

### Out of scope

The following will be closed without a fix, an advisory, or a CVE request:

- New bypasses of the SQL, Cypher, or AQL denylists: previously unlisted
  functions, alternate quoting or escaping forms (backticks, Unicode escapes,
  schema qualification), or dialect-specific syntax. See "What is, and is not,
  a security boundary" above.
- Sandbox escapes from the pandas `eval()` path when `full_eval=True`,
  including object-graph traversal via dunder attributes. `full_eval=True`
  disables the AST validator by design.
- Anything requiring `allow_dangerous_operations=True`.
- "The LLM can be prompt-injected into generating a harmful query." This is the
  documented threat model, not a defect.
- Reports that assume the agent was given a superuser or otherwise
  over-privileged database credential.
- Theoretical reports with no working proof of concept against a default
  configuration.

Reports that chain off a previous advisory as an "incomplete fix" are judged
against this same scope: if the original issue was a missing gate we will fix
it, and if it was a denylist gap it is out of scope.

## Supported versions

Security fixes ship in the latest release. Please upgrade to the current
version of `langroid` rather than requesting backports.
