"""One JSON object per log line: `ts`, `level`, `event`, plus whitelisted context fields.

Only the fields in CONTEXT_FIELDS ever leave a record, so a stray `extra=` cannot leak a
password, cookie or API key into the log. `log_config.json` wires this formatter into uvicorn.
"""

import json
import logging
from datetime import datetime, timezone

CONTEXT_FIELDS = ("user_id", "method", "path", "status", "duration_ms")

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
