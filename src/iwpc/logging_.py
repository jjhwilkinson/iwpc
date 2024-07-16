import logging
import sys


def setup_stdout_logging(level=logging.DEBUG) -> None:
    """
    Sets up the root logger to log to stdout

    Parameters
    ----------
    level
        The logging level
    """
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
