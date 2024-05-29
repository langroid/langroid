class InfiniteLoopException(Exception):
    def __init__(
        self, message: str = "Infinite loop detected", *args: object
    ) -> None: ...

class LangroidImportError(ImportError):
    def __init__(
        self,
        package: str | None = None,
        extra: str | None = None,
        error: str = "",
        *args: object
    ) -> None: ...
