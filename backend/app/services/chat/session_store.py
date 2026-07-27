"""
Session & chat history store.

Design choice: kept in its own module rather than folded into MetadataStore
-- MetadataStore is about *document* chunks, this is about *conversation*
state, a different bounded context. Both share the same SQLite engine (via
app.core.db.get_engine) so we don't open a second connection pool to the
same file, while keeping the two responsibilities in separate classes/tables.

get_recent_turns() exists specifically to feed the Phase 2 QueryRewriter --
without conversation memory, a follow-up like "what about in Europe?" has no
antecedent for the rewriter to resolve against. Only the last few turns are
returned (not the whole history) to keep the rewrite prompt small and fast.
"""

import json
import uuid
from datetime import datetime

from sqlmodel import Field, Session, SQLModel, select

from app.core.config import Settings
from app.core.db import get_engine
from app.models.schemas import Citation, HistoryMessage, MessageRole


class SessionRow(SQLModel, table=True):
    __tablename__ = "chat_sessions"
    session_id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MessageRow(SQLModel, table=True):
    __tablename__ = "chat_messages"
    message_id: str = Field(primary_key=True)
    session_id: str = Field(index=True)
    role: str
    content: str
    citations_json: str = "[]"  # serialized list[Citation]; "[]" for user messages
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatHistoryStore:
    def __init__(self, settings: Settings):
        self.engine = get_engine()
        SQLModel.metadata.create_all(self.engine)

    def ensure_session(self, session_id: str | None) -> str:
        """Returns a valid session_id, creating a new session row if none was given."""
        sid = session_id or str(uuid.uuid4())
        with Session(self.engine) as session:
            existing = session.get(SessionRow, sid)
            if not existing:
                session.add(SessionRow(session_id=sid))
                session.commit()
        return sid

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        citations: list[Citation] | None = None,
    ) -> None:
        with Session(self.engine) as session:
            session.add(
                MessageRow(
                    message_id=str(uuid.uuid4()),
                    session_id=session_id,
                    role=role.value,
                    content=content,
                    citations_json=json.dumps([c.model_dump() for c in (citations or [])]),
                )
            )
            session.commit()

    def get_history(self, session_id: str, limit: int = 50) -> list[HistoryMessage]:
        with Session(self.engine) as session:
            rows = session.exec(
                select(MessageRow)
                .where(MessageRow.session_id == session_id)
                .order_by(MessageRow.timestamp)
                .limit(limit)
            ).all()
        return [
            HistoryMessage(
                role=MessageRole(r.role),
                content=r.content,
                timestamp=r.timestamp,
                citations=[Citation(**c) for c in json.loads(r.citations_json)],
            )
            for r in rows
        ]

    def get_recent_turns(self, session_id: str, n: int = 3) -> list[tuple[str, str]]:
        """Last n (role, content) messages, oldest-first, for query-rewrite context."""
        history = self.get_history(session_id, limit=n * 2 + 2)
        turns = [(m.role.value, m.content) for m in history]
        return turns[-(n * 2):] if turns else []

    def delete_session(self, session_id: str) -> None:
        with Session(self.engine) as session:
            rows = session.exec(
                select(MessageRow).where(MessageRow.session_id == session_id)
            ).all()
            for r in rows:
                session.delete(r)
            s = session.get(SessionRow, session_id)
            if s:
                session.delete(s)
            session.commit()