"""Logging utils."""

import logging
import sys
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Self


class LogLevelError(KeyError):
    """Error type for unsupported log levels."""

    def __init__(self, s: str) -> None:
        super().__init__(f"LogLevel '{s}' doesn't exist.")


class LogLevel(Enum):
    """Log Level Enumeration."""

    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    def __str__(self) -> str:  # noqa: D105 docstring of __str__ isn't required: it must match the standard library's
        return self.name

    @classmethod
    def from_string(cls, s: str) -> Self:
        """Instantiate a LogLevel from a string."""
        try:
            return cls[s]
        except KeyError as exc:
            raise LogLevelError(s) from exc


def log_uncaught_exceptions() -> None:
    """Make all uncaught exception to be logged by the default logger.

    Keyboard exceptions and children classes are not logged so one can kill the program with ctr+C.
    """

    def handle_exception(
        exc_type: type[BaseException], exc_value: BaseException, exc_traceback: TracebackType | None
    ) -> None:
        if not issubclass(exc_type, KeyboardInterrupt):
            logger = logging.getLogger(__name__)
            logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception


def init_logging(log_file: Path | None, log_level: LogLevel) -> None:
    """(Re-)initialize all loggers.

    Configure the logger to `log_file` if it is not None, otherwise logging will go to standard output/error.
    Warnings and uncaught errors will also be automatically be logged.

    Parameters
    ----------
    log_file : Path | None
        Filename to write the log to. Default is None, in which case the log goes to standard output/error.
    log_level : str | None
        Logging level. Default is None, in which case the default of `logging` is used (WARNING).
    """
    logging.captureWarnings(True)  # log all warnings from the warnings module.  # noqa: FBT003 API is not ours
    log_uncaught_exceptions()  # log all uncaught exceptions as well

    logging_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)s:%(funcName)s %(message)s"
    handlers = [logging.FileHandler(log_file)] if log_file is not None else [logging.StreamHandler()]
    logging.basicConfig(
        level=log_level.value,
        format=logging_format,
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging configured - start logging")
