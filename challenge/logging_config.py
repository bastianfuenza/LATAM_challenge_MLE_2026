import logging

LOGGER_NAME = 'challenge'
FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the package logger.

    Args:
        level (int): minimum level emitted by the package loggers.
    """
    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(FORMAT))

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
