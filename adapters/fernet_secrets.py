"""The secret box, sealing with Fernet under a key the operator holds.

The key comes from `CAUSA_SECRET_KEY` and never from the database: a key stored
beside what it seals protects nothing from whoever can read the database, and
reading the database is what sealing is for. Generate one with

    python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Without it the platform runs and cannot store a new secret; a secret sealed
earlier cannot be opened, and the resource says so.
"""
from __future__ import annotations

import base64
import os

from core.secrets import SecretUnavailable


class FernetSecretBox:
    def __init__(self, key: str | None = None) -> None:
        key = key if key is not None else os.getenv("CAUSA_SECRET_KEY", "")
        self._fernet = None
        self.reason = ""
        if not key:
            self.reason = "CAUSA_SECRET_KEY is not set"
        else:
            try:
                from cryptography.fernet import Fernet
                self._fernet = Fernet(key.encode())
            except ValueError:
                self.reason = "CAUSA_SECRET_KEY is not a Fernet key"
        self.available = self._fernet is not None

    def seal(self, plain: str) -> str:
        if self._fernet is None:
            raise SecretUnavailable(f"cannot seal a secret: {self.reason}")
        return str(self._fernet.encrypt(plain.encode()).decode())

    def open(self, token: str) -> str:
        if self._fernet is None:
            raise SecretUnavailable(f"cannot open a stored secret: {self.reason}")
        from cryptography.fernet import InvalidToken
        try:
            return str(self._fernet.decrypt(token.encode()).decode())
        except InvalidToken as exc:
            raise SecretUnavailable(
                "cannot open a stored secret: it was sealed with another key") from exc


class SecretBoxStub:
    """Reversible without a key, for tests. Not a secret box: anybody can open
    what it seals, which is exactly what a test needs and nothing else may."""

    available = True

    def seal(self, plain: str) -> str:
        return "stub:" + base64.b64encode(plain.encode()).decode()

    def open(self, token: str) -> str:
        if not token.startswith("stub:"):
            raise SecretUnavailable("cannot open a stored secret: not sealed by the stub")
        return base64.b64decode(token[5:]).decode()
