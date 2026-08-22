from enum import Enum
import json

class StreamEvent(str, Enum):
    START = "start"
    TOKEN = "token"
    CITATIONS = "citations"
    END = "end"
    ERROR = "error"

def sse_event(event: StreamEvent, data) -> str:

    return (
        f"event: {event.value}\n"
        f"data: {json.dumps(data)}\n\n"
    )