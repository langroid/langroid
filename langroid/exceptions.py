from typing import List, Optional


class XMLException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InfiniteLoopException(Exception):
    def __init__(self, message: str = "Infinite loop detected", *args: object) -> None:
        super().__init__(message, *args)


class LangroidImportError(ImportError):
    def __init__(
        self,
        package: Optional[str] = None,
        extra: Optional[str | List[str]] = None,
        error: str = "",
        *args: object,
    ) -> None:
        """
        Generate helpful warning when attempting to import package or module.

        Args:
            package (str): The name of the package to import.
            extra (str): The name of the extras package required for this import.
            error (str): The error message to display. Depending on context, we
                can set this by capturing the ImportError message.

        """
        if error == "" and package is not None:
            error = f"{package} is not installed by default with Langroid.\n"

        if extra:
            if isinstance(extra, list):
                help_preamble = f"""
                If you want to use it, please install langroid with one of these 
                extras: {', '.join(extra)}. The examples below use the first one, 
                i.e. {extra[0]}.
                """
                extra = extra[0]
            else:
                help_preamble = f"""
                If you want to use it, please install langroid with the
                `{extra}` extra.
                """

            install_help = f"""
                {help_preamble}
                
                If you are using pip:
                pip install "langroid[{extra}]"
                
                For multiple extras, you can separate them with commas:
                pip install "langroid[{extra},another-extra]"
                
                If you are using Poetry:
                poetry add langroid --extras "{extra}"
                
                For multiple extras with Poetry, list them with spaces:
                poetry add langroid --extras "{extra} another-extra"

                If you are using uv:
                uv add "langroid[{extra}]"

                For multiple extras with uv, you can separate them with commas: 
                uv add "langroid[{extra},another-extra]"
                
                If you are working within the langroid dev env (which uses uv),
                you can do:
                uv sync --dev --extra "{extra}"
                or if you want to include multiple extras:
                uv sync --dev --extra "{extra}" --extra "another-extra"
                """
        else:
            install_help = """
                If you want to use it, please install it in the same
                virtual environment as langroid.
                """
        msg = error + install_help

        super().__init__(msg, *args)
