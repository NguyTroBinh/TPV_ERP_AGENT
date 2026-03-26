"""
memory/session_store.py
───────────────────────
Redis-backed session + conversation history.

Key structure:
  chatbot:meta:{session_id}              -> Hash  (session metadata)
  chatbot:history:{session_id}           -> List  (message JSONs, RPUSH each)
  chatbot:tenant:{tenant_id}:sessions    -> Set   (session IDs for sidebar)

Design:
- Moi message RPUSH rieng -> khong can re-serialize toan bo session
- Meta luu rieng Hash -> doc nhanh cho sidebar (khong load messages)
- TTL tu dong refresh moi khi user active
- ULID cho session ID (time-sortable, unique)
"""

import json
import structlog
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field

from redis.asyncio import Redis, ConnectionPool
from config import get_settings
from config.constants import Role

logger = structlog.get_logger(__name__)
settings = get_settings()

SESSION_TTL = settings.redis.session_ttl  # default 3600s


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATA MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Message:
    role: str           # "user" | "assistant" | "tool"
    content: str
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_context: Optional[str] = None  # Condensed tool results — cho follow-up questions
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_llm_format(self) -> dict:
        """Convert to OpenAI message format for LLM."""
        return {"role": self.role, "content": self.content}


@dataclass
class Session:
    session_id: str
    user_id: str
    tenant_id: str
    title: str = "New Chat"
    mode: str = "chat"
    messages: list[Message] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SESSION STORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SessionStore:
    """
    Redis-backed session management.

    Atomic operations via pipeline:
    - create: HSET meta + EXPIRE + SADD tenant set
    - add_message: RPUSH history + EXPIRE + HSET updated_at
    - get_history: LRANGE history list
    """

    def __init__(self):
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[Redis] = None

    # ── Connection ─────────────────────────────────────────

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._pool = ConnectionPool.from_url(
            settings.redis.url,
            max_connections=20,
            decode_responses=True,
        )
        self._client = Redis(connection_pool=self._pool)
        await self._client.ping()
        logger.info("session_store.connected")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._pool:
            await self._pool.disconnect()

    @property
    def client(self) -> Redis:
        if self._client is None:
            raise RuntimeError("SessionStore chua connect(). Goi await session_store.connect() truoc.")
        return self._client

    # ── Key helpers ────────────────────────────────────────

    @staticmethod
    def _key_meta(session_id: str) -> str:
        return f"chatbot:meta:{session_id}"

    @staticmethod
    def _key_history(session_id: str) -> str:
        return f"chatbot:history:{session_id}"

    @staticmethod
    def _key_tenant_sessions(tenant_id: str) -> str:
        return f"chatbot:tenant:{tenant_id}:sessions"

    # ── Session CRUD ──────────────────────────────────────

    async def create_session(
        self,
        tenant_id: str,
        user_id: str = "anonymous",
        title: str = "New Chat",
        mode: str = "chat",
    ) -> str:
        """
        Tao session moi, luu Meta vao Redis Hash.
        Returns session_id (ULID — time-sortable).
        """
        session_id = self._generate_id()
        now = datetime.now(timezone.utc).isoformat()

        meta = {
            "id": session_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "title": title,
            "mode": mode,
            "created_at": now,
            "updated_at": now,
        }

        meta_key = self._key_meta(session_id)
        tenant_key = self._key_tenant_sessions(tenant_id)

        pipe = self.client.pipeline()
        pipe.hset(meta_key, mapping=meta)
        pipe.expire(meta_key, SESSION_TTL)
        pipe.sadd(tenant_key, session_id)
        await pipe.execute()

        logger.info("session.created", session_id=session_id, tenant=tenant_id)
        return session_id

    async def get_or_create(
        self,
        session_id: Optional[str],
        user_id: str,
        tenant_id: str,
    ) -> Session:
        """
        Lay session neu con song, tao moi neu khong co.
        Giu nguyen session_id tu frontend de duy tri bo nho.
        """
        if session_id:
            # Pipeline: HGETALL + EXPIRE×2 trong 1 roundtrip
            pipe = self.client.pipeline()
            pipe.hgetall(self._key_meta(session_id))
            pipe.expire(self._key_meta(session_id), SESSION_TTL)
            pipe.expire(self._key_history(session_id), SESSION_TTL)
            results = await pipe.execute()
            meta = results[0]
            if meta:
                return Session(
                    session_id=session_id,
                    user_id=meta.get("user_id", user_id),
                    tenant_id=meta.get("tenant_id", tenant_id),
                    title=meta.get("title", "New Chat"),
                    mode=meta.get("mode", "chat"),
                    created_at=meta.get("created_at", ""),
                    updated_at=meta.get("updated_at", ""),
                )

        # Create new session
        new_id = session_id or self._generate_id()
        now = datetime.now(timezone.utc).isoformat()

        meta = {
            "id": new_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "title": "New Chat",
            "mode": "chat",
            "created_at": now,
            "updated_at": now,
        }

        meta_key = self._key_meta(new_id)
        tenant_key = self._key_tenant_sessions(tenant_id)

        pipe = self.client.pipeline()
        pipe.hset(meta_key, mapping=meta)
        pipe.expire(meta_key, SESSION_TTL)
        pipe.sadd(tenant_key, new_id)
        await pipe.execute()

        logger.info("session.created", session_id=new_id, user_id=user_id)
        return Session(
            session_id=new_id,
            user_id=user_id,
            tenant_id=tenant_id,
            created_at=now,
            updated_at=now,
        )

    async def session_exists(self, session_id: str) -> bool:
        """Kiem tra session con ton tai (chua het han) khong."""
        return await self.client.exists(self._key_meta(session_id)) > 0

    async def get(self, session_id: str) -> Optional[Session]:
        """
        Load session voi TAT CA messages.
        Dung khi can full history (vd: load chat cu).
        """
        meta = await self.client.hgetall(self._key_meta(session_id))
        if not meta:
            return None

        # Load all messages from Redis List
        raw_msgs = await self.client.lrange(self._key_history(session_id), 0, -1)
        messages = []
        for raw in raw_msgs:
            try:
                data = json.loads(raw)
                messages.append(Message(
                    role=data["role"],
                    content=data["content"],
                    tool_name=data.get("tool_name"),
                    tool_call_id=data.get("tool_call_id"),
                    tool_context=data.get("tool_context"),
                    created_at=data.get("created_at", ""),
                ))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("session.corrupted_message", session_id=session_id, error=str(e), raw=raw[:200])
                continue

        return Session(
            session_id=session_id,
            user_id=meta.get("user_id", ""),
            tenant_id=meta.get("tenant_id", ""),
            title=meta.get("title", "New Chat"),
            mode=meta.get("mode", "chat"),
            messages=messages,
            created_at=meta.get("created_at", ""),
            updated_at=meta.get("updated_at", ""),
        )

    async def delete(self, session_id: str) -> None:
        """Xoa session: meta + history + remove khoi tenant set."""
        meta = await self.client.hgetall(self._key_meta(session_id))
        tenant_id = meta.get("tenant_id")

        pipe = self.client.pipeline()
        pipe.delete(self._key_meta(session_id))
        pipe.delete(self._key_history(session_id))
        if tenant_id:
            pipe.srem(self._key_tenant_sessions(tenant_id), session_id)
        await pipe.execute()

        logger.info("session.deleted", session_id=session_id)

    # ── Message operations ────────────────────────────────

    async def add_message(
        self,
        session: Session,
        role: str,
        content: str,
        **kwargs,
    ) -> Message:
        """
        RPUSH message vao Redis List + refresh TTL.
        Moi message duoc luu rieng le — khong re-serialize toan bo session.
        """
        msg = Message(role=role, content=content, **kwargs)
        sid = session.session_id

        serialized = json.dumps({
            "role": msg.role,
            "content": msg.content,
            "tool_name": msg.tool_name,
            "tool_call_id": msg.tool_call_id,
            "tool_context": msg.tool_context,
            "created_at": msg.created_at,
        }, ensure_ascii=False)

        now = datetime.now(timezone.utc).isoformat()
        meta_key = self._key_meta(sid)
        hist_key = self._key_history(sid)

        pipe = self.client.pipeline()
        pipe.rpush(hist_key, serialized)           # [0] = new length
        pipe.expire(hist_key, SESSION_TTL)
        pipe.expire(meta_key, SESSION_TTL)
        pipe.hset(meta_key, "updated_at", now)
        results = await pipe.execute()

        # Auto-set title từ first user message (dùng length trả về từ RPUSH)
        if role == Role.USER and results[0] == 1:
            await self.client.hset(meta_key, "title", content[:50])

        # Keep in-memory copy in sync
        session.messages.append(msg)
        session.updated_at = now

        return msg

    async def delete_last_message(self, session_id: str) -> None:
        """Xoa tin nhan cuoi cung (Rollback khi LLM loi)."""
        await self.client.rpop(self._key_history(session_id))

    async def get_history(
        self,
        session: Session,
        last_n: Optional[int] = None,
    ) -> list[dict]:
        """
        Lay chat history cho LLM (tu Redis, khong tu in-memory).
        Chi tra user/assistant messages.
        """
        hist_key = self._key_history(session.session_id)
        raw_msgs = await self.client.lrange(hist_key, 0, -1)

        # Filter user/assistant, inject tool_context cho follow-up questions
        chat_msgs = []
        for raw in raw_msgs:
            try:
                data = json.loads(raw)
                if data["role"] in (Role.USER, Role.ASSISTANT):
                    content = data["content"]
                    # Inject tool context — LLM thấy dữ liệu đã tra cứu ở lượt trước
                    if data["role"] == Role.ASSISTANT and data.get("tool_context"):
                        content = (
                            f"<retrieved_data>\n{data['tool_context']}\n</retrieved_data>"
                            f"\n\n{content}"
                        )
                    chat_msgs.append({
                        "role": data["role"],
                        "content": content,
                    })
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("session.corrupted_history", error=str(e), raw=raw[:200])
                continue

        if last_n:
            chat_msgs = chat_msgs[-last_n:]

        return chat_msgs

    async def get_messages(self, session_id: str, limit: int = 50) -> list[dict]:
        """
        Lay raw messages theo session_id.
        Compatible voi pattern cu (memory_mgr).
        """
        hist_key = self._key_history(session_id)
        start = -limit if limit > 0 else 0
        raw_msgs = await self.client.lrange(hist_key, start, -1)

        results = []
        for raw in raw_msgs:
            try:
                results.append(json.loads(raw))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("session.corrupted_raw_message", error=str(e), raw=raw[:200])
                continue
        return results

    # ── Convenience wrappers (agent.py dung) ──────────────

    async def add_user_message(self, session: Session, content: str) -> Message:
        return await self.add_message(session, role=Role.USER, content=content)

    async def add_assistant_message(self, session: Session, content: str) -> Message:
        return await self.add_message(session, role=Role.ASSISTANT, content=content)

    async def add_tool_result(
        self,
        session: Session,
        tool_name: str,
        content: str,
        tool_call_id: str,
    ) -> Message:
        return await self.add_message(
            session, role=Role.TOOL, content=content,
            tool_name=tool_name, tool_call_id=tool_call_id,
        )

    # ── Tenant session management (sidebar) ───────────────

    async def get_sessions_by_tenant(self, tenant_id: str) -> list[dict]:
        """
        Lay danh sach sessions cho sidebar.
        Auto-cleanup session da het han (meta key expired).
        """
        tenant_key = self._key_tenant_sessions(tenant_id)
        session_ids = await self.client.smembers(tenant_key)
        if not session_ids:
            return []

        results = []
        for sid in session_ids:
            meta = await self.client.hgetall(self._key_meta(sid))
            if meta:
                results.append({
                    "chatSessionId": meta.get("id", sid),
                    "title": meta.get("title", "New Chat"),
                    "createTime": meta.get("created_at", ""),
                    "mode": meta.get("mode", "chat"),
                })
            else:
                # Meta expired -> don dep khoi tenant set
                await self.client.srem(tenant_key, sid)

        # Moi nhat len dau (ULID sortable theo thoi gian)
        results.sort(key=lambda x: x.get("chatSessionId", ""), reverse=True)
        return results

    # ── Health ────────────────────────────────────────────

    async def health_check(self) -> bool:
        if self._client is None:
            await self.connect()
        await self._client.ping()
        return True

    # ── Private ───────────────────────────────────────────

    async def _refresh_ttl(self, session_id: str) -> None:
        """Gia han TTL cho ca meta va history."""
        pipe = self.client.pipeline()
        pipe.expire(self._key_meta(session_id), SESSION_TTL)
        pipe.expire(self._key_history(session_id), SESSION_TTL)
        await pipe.execute()

    @staticmethod
    def _generate_id() -> str:
        """Generate time-sortable ID (ULID preferred, UUID4 fallback)."""
        try:
            import ulid
            return ulid.ulid()
        except (ImportError, AttributeError):
            import uuid
            return str(uuid.uuid4())

    # ── Legacy compat ─────────────────────────────────────

    async def save(self, session: Session) -> None:
        """No-op — messages duoc luu rieng le qua add_message."""
        pass

    async def extend_ttl(self, session_id: str) -> None:
        await self._refresh_ttl(session_id)


# Singleton
session_store = SessionStore()
