from __future__ import annotations
import json
import secrets
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class Pending:
    token: str
    user_id: int
    inn: str
    email: str


class Storage:
    def __init__(self, base_dir: str) -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.pending_path = self.base / "pending.json"
        self.users_path = self.base / "users.json"
        self.history_dir = self.base / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        if not self.pending_path.exists():
            self.pending_path.write_text("{}", encoding="utf-8")
        if not self.users_path.exists():
            self.users_path.write_text("{}", encoding="utf-8")

    def _read_json(self, path: Path) -> Dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def create_pending(self, user_id: int, inn: str, email: str) -> Pending:
        token = secrets.token_urlsafe(16)
        pending = Pending(token=token, user_id=user_id, inn=inn, email=email)
        data = self._read_json(self.pending_path)
        data[token] = asdict(pending)
        self._write_json(self.pending_path, data)
        return pending

    def pop_pending(self, token: str) -> Optional[Pending]:
        data = self._read_json(self.pending_path)
        info = data.pop(token, None)
        if info is not None:
            self._write_json(self.pending_path, data)
            return Pending(**info)
        return None

    def mark_authorized(self, user_id: int, inn: str, email: str) -> None:
        users = self._read_json(self.users_path)
        users[str(user_id)] = {"inn": inn, "email": email}
        self._write_json(self.users_path, users)

    def get_user(self, user_id: int) -> Optional[Dict[str, str]]:
        users = self._read_json(self.users_path)
        return users.get(str(user_id))

    def is_authorized(self, user_id: int) -> bool:
        return self.get_user(user_id) is not None

    # История диалога для ИИ
    def _history_path(self, user_id: int) -> Path:
        return self.history_dir / f"{user_id}.json"

    def get_history(self, user_id: int, limit: int = 20) -> List[Dict[str, str]]:
        path = self._history_path(user_id)
        if not path.exists():
            return []
        try:
            all_items = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(all_items, list):
                return []
            return all_items[-limit:]
        except Exception:
            return []

    def append_history(self, user_id: int, role: str, content: str) -> None:
        path = self._history_path(user_id)
        try:
            items: List[Dict[str, str]] = []
            if path.exists():
                items = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(items, list):
                    items = []
        except Exception:
            items = []
        items.append({"role": role, "content": content})
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8") 