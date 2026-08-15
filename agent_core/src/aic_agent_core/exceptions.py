class QueryRouterError(RuntimeError):
    """Base error raised by the query router."""


class EmptyQueryError(QueryRouterError, ValueError):
    """Raised when the raw query is empty or only whitespace."""


class RouterConfigurationError(QueryRouterError):
    """Raised when Gemini credentials or settings are invalid."""


class StructuredOutputError(QueryRouterError):
    """Raised when Gemini repeatedly returns an invalid structured result."""

