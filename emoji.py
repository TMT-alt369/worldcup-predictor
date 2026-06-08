"""Local fallback for environments that import `emoji` without the package.

The app stores flag symbols directly as Unicode text and does not require the
external emoji package. This tiny compatibility layer prevents deployment
failures if a stale cloud build or optional path imports `emoji`.
"""

__version__ = "local-fallback"


def emojize(text: str, *args, **kwargs) -> str:
    return text


def demojize(text: str, *args, **kwargs) -> str:
    return text


def is_emoji(value: str) -> bool:
    return bool(value)
