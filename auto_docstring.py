from pathlib import Path
import mkdocs_gen_files

# -----------------------------------------------------#
#                    Configuration                    #
# -----------------------------------------------------#
src_dir = "langroid"
repo_root = "https://github.com/langroid/langroid/tree/main/"
nav = mkdocs_gen_files.Nav()

# -----------------------------------------------------#
#                       Runner                        #
# -----------------------------------------------------#
""" Generate code reference pages and navigation

    Based on the recipe of mkdocstrings:
    https://github.com/mkdocstrings/mkdocstrings

    Credits:
    Timothée Mazzucotelli
    https://github.com/pawamoy
"""
# Iterate over each Python file
for path in sorted(Path(src_dir).rglob("*.py")):
    if ".ipynb_checkpoints" in str(path):
        continue

    # Get path in module, documentation and absolute
    module_path = path.relative_to(src_dir).with_suffix("")
    doc_path = path.relative_to(src_dir).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    # Handle edge cases
    parts = (src_dir,) + tuple(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    nav[parts] = doc_path.as_posix()

    # Write docstring documentation to disk via parser
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        full_code_path = repo_root + "/" + str(path)
        fd.write(f"[{path}]({full_code_path})\n")
        fd.write(f"::: {ident}")
    # Update parser
    mkdocs_gen_files.set_edit_path(full_doc_path, path)
    print(f"Doing docs for {full_doc_path}, {path}")

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
