from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class QueryIn(BaseModel):
    app_id: str
    user_id: str
    session_id: str = "default"
    query: str
    use_history: bool = True
    history_turns: int = Field(5, ge=0, le=20)
    overrides: Optional[Dict[str, Any]] = None