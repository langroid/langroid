!!! note
Caveat Lector. May not be fully accurate. Trust but Verify!

Splitting the chunk in the middle might not respect syntactic
structure. To better handle this, we can adjust the `_split_chunk` function to
split at the nearest token boundary instead of directly splitting at the middle
of the chunk. Here's an updated implementation:

```python
from typing import Callable, List
from pygments import lex
from pygments.token import Token
from pygments.lexers import get_lexer_by_name


def chunk_code(
        code: str, language: str, max_tokens: int, len_fn: Callable[[str], int]
) -> List[str]:
    def _split_chunk(chunk: str, lexer) -> List[str]:
        if len_fn(chunk) <= max_tokens:
            return [chunk]

        chunk_tokens = list(lex(chunk, lexer))

        # Find the nearest token boundary that splits chunk into smaller pieces
        left_half_tokens = []
        right_half_tokens = []

        left_len = 0
        for token_type, token_value in chunk_tokens:
            token_len = len_fn(token_value)
            if left_len + token_len <= max_tokens:
                left_half_tokens.append((token_type, token_value))
                left_len += token_len
            else:
                right_half_tokens = chunk_tokens[len(left_half_tokens):]
                break

        left_half = "".join(token_value for _, token_value in left_half_tokens)
        right_half = "".join(
            token_value for _, token_value in right_half_tokens)

        return _split_chunk(left_half, lexer) + _split_chunk(right_half, lexer)

    lexer = get_lexer_by_name(language)
    tokens = list(lex(code, lexer))

    chunks = []
    current_chunk = ""
    for token_type, token_value in tokens:
        if token_type in Token.Text.Whitespace:
            current_chunk += token_value
        else:
            token_tokens = len_fn(token_value)
            if len_fn(current_chunk) + token_tokens <= max_tokens:
                current_chunk += token_value
            else:
                chunks.extend(_split_chunk(current_chunk, lexer))
                current_chunk = token_value

    if current_chunk:
        chunks.extend(_split_chunk(current_chunk, lexer))

    return chunks
```

Now, the `_split_chunk` function takes an additional `lexer` argument and uses
it to split the chunk at the nearest token boundary, which should respect the
syntactic structure better.

# Chunking with `asttokens`

You're right, manually handling the splitting process can be tedious. A possible
alternative is to use a package like `asttokens`, which can parse Python code
into an Abstract Syntax Tree (AST) and provide token information. This will help
you split the code more accurately by working with the AST structure.

To use `asttokens`, you will first need to install it:

```bash
pip install asttokens
```

Here's an example implementation using `asttokens` to split Python code while
respecting the hierarchical structure:

```python
import ast
from typing import Callable, List
from asttokens import ASTTokens


def split_ast(node, max_tokens: int, len_fn: Callable[[str], int]):
    if not isinstance(node, ast.AST):
        return []

    chunks = []
    chunk = ""

    for child_node in ast.iter_child_nodes(node):
        child_source = asttokens.unparse(child_node)

        if len_fn(chunk) + len_fn(child_source) > max_tokens:
            if chunk:
                chunks.append(chunk)
            chunk = child_source
        else:
            if chunk:
                chunk += "\n"
            chunk += child_source

        chunks.extend(split_ast(child_node, max_tokens, len_fn))

    if chunk:
        chunks.append(chunk)

    return chunks


def chunk_code(code: str, max_tokens: int, len_fn: Callable[[str], int]) ->


    List[str]:
atok = ASTTokens(code, parse=True)
tree = atok.tree

return split_ast(tree, max_tokens, len_fn)
```

Note that this implementation is specific to Python code. For other languages,
you would need to find a package that can parse the code into a similar tree
structure.

Keep in mind that AST-based splitting might not always produce syntactically
correct code in every case, since the produced chunks could be missing necessary
imports or declarations from other parts of the code. However, it should
generally respect the hierarchical structure better than the previous approach.

# Chunking with `tree-sitter`

There isn't a universal language-agnostic tool that can parse any programming
language into an abstract syntax tree, as each language has its own syntax and
semantics. However, there are tools and libraries designed to work with multiple
languages.

One such tool is Tree-sitter, a language parsing library that generates syntax
trees for several programming languages. Tree-sitter has a Python binding, which
you can use to parse code for different languages.

To install Tree-sitter's Python binding:

```bash
pip install tree-sitter
```

To parse code with Tree-sitter, you'll need to download the appropriate grammar
for the language you're working with. For example, if you want to parse Python
code, you would need the Tree-sitter Python grammar:

```bash
git clone https://github.com/tree-sitter/tree-sitter-python
```

Here's an example implementation using Tree-sitter to chunk Python code:

```python
import os
from typing import Callable, List
from tree_sitter import Language, Parser

# Set the path to the compiled grammar of the language you want to parse
TREE_SITTER_PYTHON_PATH = os.path.abspath("tree-sitter-python")

PYTHON_LANGUAGE = Language(TREE_SITTER_PYTHON_PATH, "python")
parser = Parser()
parser.set_language(PYTHON_LANGUAGE)


def split_tree_sitter(node, max_tokens: int, len_fn: Callable[[str], int],
                      source_code: str):
    if node.child_count == 0:
        return []

    chunks = []
    chunk = ""

    for child_node in node.children:
        start = child_node.start_byte
        end = child_node.end_byte
        child_source = source_code[start:end]

        if len_fn(chunk) + len_fn(child_source) > max_tokens:
            if chunk:
                chunks.append(chunk)
            chunk = child_source
        else:
            if chunk:
                chunk += "\n"
            chunk += child_source

        chunks.extend(
            split_tree_sitter(child_node, max_tokens, len_fn, source_code))

    if chunk:
        chunks.append(chunk)

    return chunks


def chunk_code(code: str, max_tokens: int, len_fn: Callable[[str], int]) ->
List[str]:
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    return split_tree_sitter(root_node, max_tokens, len_fn, code)
```

Note that this example is specific to Python code. To work with other languages,
you would need to download the appropriate Tree-sitter grammar for that language
and adjust the language setup accordingly.

Tree-sitter can parse multiple programming languages, but it is not completely
language-agnostic, as it still requires specific grammars for each language.
However, it should provide better support for multiple languages compared to
using language-specific libraries like `asttokens`.
