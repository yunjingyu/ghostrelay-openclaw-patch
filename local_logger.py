import logging
import sys
from pathlib import Path


_configured = False


def _configure_root_logger() -> None:
    global _configured
    if _configured:
        return

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_dir / "ghostrelay.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger("ghostrelay")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    return logging.getLogger(f"ghostrelay.{name}")

