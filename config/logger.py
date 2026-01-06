"""Logging configuration."""
from loguru import logger
import sys
from .settings import settings


def setup_logger():
    """Configure application logging."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
    )
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="10 days",
        level=settings.log_level,
    )
    return logger


# Initialize logger
setup_logger()







