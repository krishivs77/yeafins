"""Structured API exceptions."""

from __future__ import annotations


class ApiError(Exception):
    """An error safe to return to an API client."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class ServiceUnavailableError(ApiError):
    """The inference service is not currently ready."""

    def __init__(self, message: str = "The chess engine is currently unavailable.") -> None:
        super().__init__(503, "service_unavailable", message)
