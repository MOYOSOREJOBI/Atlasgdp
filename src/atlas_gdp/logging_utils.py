from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any


def get_logger(name: str = "atlas_gdp") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {"ts": datetime.now(UTC).isoformat(), "event": event, **fields}
    logger.info(json.dumps(payload, default=str))
