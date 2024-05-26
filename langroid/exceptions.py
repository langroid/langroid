class InfiniteLoopException(Exception):
    def __init__(self, message: str = "Infinite loop detected", *args: object) -> None:
        super().__init__(message, *args)
