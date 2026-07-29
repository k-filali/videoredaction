"""Protect the append-only audit log.

Revision ID: b4e71cc82a16
Revises: d9057289416c
Create Date: 2026-07-28
"""

from collections.abc import Sequence

from alembic import op

revision: str = "b4e71cc82a16"
down_revision: str | None = "d9057289416c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "sqlite":
        return
    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS prevent_review_action_update
        BEFORE UPDATE ON review_actions
        BEGIN
            SELECT RAISE(ABORT, 'review actions are append-only');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS prevent_review_action_delete
        BEFORE DELETE ON review_actions
        BEGIN
            SELECT RAISE(ABORT, 'review actions are append-only');
        END
        """
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "sqlite":
        return
    op.execute("DROP TRIGGER IF EXISTS prevent_review_action_delete")
    op.execute("DROP TRIGGER IF EXISTS prevent_review_action_update")

