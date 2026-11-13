"""Explicit logging setup; importing RUTAM never changes root logging."""

import logging


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("rutam")
    logger.setLevel(level)
    if not any(getattr(handler, "_rutam", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        handler._rutam = True
        logger.addHandler(handler)
    logger.propagate = False
    return logger
