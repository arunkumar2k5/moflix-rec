"""One JSON object per log line: `ts`, `level`, `event`, plus whitelisted context fields.

Only the fields of RequestContext ever leave a record, so a stray `extra=` cannot leak a
password, cookie or API key into the log. `log_config.json` wires this formatter into uvicorn.
"""

import json
import logging
from datetime import datetime, timezone
from typing import TypedDict


class RequestContext(TypedDict):
    method: str
    path: str
    status: int
    duration_ms: float
    user_id: int | None


CONTEXT_FIELDS = tuple(RequestContext.__annotations__)

logger = logging.getLogger("moflix")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        for name in CONTEXT_FIELDS:
            value = getattr(record, name, None)
            if value is not None:
                line[name] = value
        if record.exc_info:
            line["exc"] = self.formatException(record.exc_info)
        return json.dumps(line)
