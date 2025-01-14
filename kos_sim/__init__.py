import logging

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Get logger for this package
logger = logging.getLogger("kos_sim")

__version__ = "0.0.1"
