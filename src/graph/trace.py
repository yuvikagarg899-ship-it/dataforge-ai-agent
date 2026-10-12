from datetime import timezone ,datetime
def trace_event(state:dict,node:str,summary:str) -> list[dict]:
    """Return the full updated trace list,
    PipelineState has no reduced for trace , so each node must return the whole list (old events plus the new one),not just the new event.
    """
    event = {
        "node":node,
        "summary":summary,
        "at":datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return state.get("trace",[])+[event]
