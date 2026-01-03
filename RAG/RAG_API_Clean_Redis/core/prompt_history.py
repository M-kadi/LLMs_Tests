from typing import List, Dict, Any

def format_history_for_prompt(turns: List[Dict[str, Any]]) -> str:
    if not turns:
        return ""
    turns = list(reversed(turns))
    return "\n\n".join(
        f"[{t['ts']}\nUser: {t['q']}\nAssistant: {t['a']}]" for t in turns
    )