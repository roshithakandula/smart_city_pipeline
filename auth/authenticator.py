"""
Authentication and role-based access control (RBAC).

Author: Roshitha K

* Passwords are hashed with bcrypt (per-password random salt).
* Sessions are stateless JWTs (HS256) carrying `sub`, `role`, `iat`, `exp`.
* Page access is decided by a single ROLE_PAGES table so the UI and any
  server-side check share one source of truth.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import bcrypt
import jwt

import config

log = logging.getLogger(__name__)

ROLE_ADMIN = "admin"
ROLE_USER = "user"

PAGE_DASHBOARD = "Executive Analytics"
PAGE_PREDICT = "PyTorch Predictions"
PAGE_GENAI = "GenAI Assistant"
PAGE_DB = "Database Management"

ROLE_PAGES: Dict[str, tuple] = {
    ROLE_ADMIN: (PAGE_DASHBOARD, PAGE_PREDICT, PAGE_GENAI, PAGE_DB),
    ROLE_USER: (PAGE_DASHBOARD, PAGE_PREDICT, PAGE_GENAI),
}

# Demo accounts required by the project brief. Change before any real use.
DEFAULT_ACCOUNTS = {
    "admin": ("admin123", ROLE_ADMIN),
    "analyst": ("user123", ROLE_USER),
}

MAX_PASSWORD_BYTES = 72  # bcrypt hard limit


def hash_password(password: str, rounds: int = config.BCRYPT_ROUNDS) -> str:
    raw = password.encode("utf-8")
    if not raw:
        raise ValueError("Password must not be empty.")
    if len(raw) > MAX_PASSWORD_BYTES:
        raise ValueError(f"Password exceeds bcrypt's {MAX_PASSWORD_BYTES}-byte limit.")
    return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=rounds)).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    raw = password.encode("utf-8")
    if not raw or len(raw) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(raw, hashed.encode("ascii"))
    except ValueError:
        return False


@dataclass(frozen=True)
class AuthResult:
    ok: bool
    message: str
    username: Optional[str] = None
    role: Optional[str] = None
    token: Optional[str] = None


class Authenticator:
    """In-memory user store with bcrypt hashes, JWT issuing and RBAC checks."""

    def __init__(self, accounts: Optional[dict] = None) -> None:
        accounts = accounts or DEFAULT_ACCOUNTS
        self._users: Dict[str, dict] = {
            name: {"hash": hash_password(pw), "role": role}
            for name, (pw, role) in accounts.items()
        }
        # Used to burn equal time for unknown usernames (timing side-channel).
        self._dummy_hash = hash_password("not-a-real-password")

    # -- credentials ---------------------------------------------------------
    def authenticate(self, username: str, password: str, requested_role: str) -> AuthResult:
        username = (username or "").strip().lower()
        user = self._users.get(username)
        if user is None:
            verify_password(password or "x", self._dummy_hash)
            return AuthResult(False, "Invalid username or password.")
        if not verify_password(password or "", user["hash"]):
            return AuthResult(False, "Invalid username or password.")
        if requested_role == ROLE_ADMIN and user["role"] != ROLE_ADMIN:
            return AuthResult(False, "This account does not have the Admin role.")
        # An admin may deliberately sign in with the lower-privilege User view.
        effective_role = requested_role if requested_role in ROLE_PAGES else user["role"]
        token = self.create_token(username, effective_role)
        return AuthResult(True, "Login successful.", username, effective_role, token)

    # -- tokens --------------------------------------------------------------
    def create_token(self, username: str, role: str) -> str:
        now = int(time.time())
        payload = {
            "sub": username,
            "role": role,
            "iat": now,
            "exp": now + config.JWT_EXPIRE_MINUTES * 60,
        }
        return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)

    def verify_token(self, token: Optional[str]) -> Optional[dict]:
        if not token:
            return None
        try:
            claims = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
        except jwt.PyJWTError as exc:
            log.info("token rejected: %s", exc)
            return None
        if claims.get("role") not in ROLE_PAGES or claims.get("sub") not in self._users:
            return None
        return claims

    # -- RBAC ----------------------------------------------------------------
    @staticmethod
    def allowed_pages(role: str) -> tuple:
        return ROLE_PAGES.get(role, ())

    @staticmethod
    def can_access(role: str, page: str) -> bool:
        return page in ROLE_PAGES.get(role, ())
