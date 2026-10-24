"""One JSON object per log line: `ts`, `level`, `event`, and `user_id` when the call supplies it.

Only the message and `extra={"user_id": ...}` reach the line, so callers must never put a
password, cookie or API key in the message.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        user_id = getattr(record, "user_id", None)
        if user_id is not None:
            line["user_id"] = user_id
        if record.exc_info:
            line["exc"] = self.formatException(record.exc_info)
        return json.dumps(line)


# Replaces uvicorn's default config so its server and access logs are JSON lines too.
LOG_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"json": {"()": JsonFormatter}},
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {"handlers": ["stdout"], "level": "INFO"},
}
