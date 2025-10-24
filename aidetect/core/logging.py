import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    Configures a basic structured logger.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # Log to stderr to keep stdout clean for reports
    )
    logging.getLogger().name = "aidetect"
