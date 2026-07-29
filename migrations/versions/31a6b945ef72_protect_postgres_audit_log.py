"""Protect the PostgreSQL append-only audit log.

Revision ID: 31a6b945ef72
Revises: c7b28f914d62
Create Date: 2026-07-29
"""

from collections.abc import Sequence

from alembic import op

revision: str = "31a6b945ef72"
down_revision: str | None = "c7b28f914d62"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        """
        CREATE FUNCTION prevent_review_action_mutation()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION 'review actions are append-only';
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER prevent_review_action_update
        BEFORE UPDATE ON review_actions
        FOR EACH ROW EXECUTE FUNCTION prevent_review_action_mutation()
        """
    )
    op.execute(
        """
        CREATE TRIGGER prevent_review_action_delete
        BEFORE DELETE ON review_actions
        FOR EACH ROW EXECUTE FUNCTION prevent_review_action_mutation()
        """
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP TRIGGER IF EXISTS prevent_review_action_delete ON review_actions")
    op.execute("DROP TRIGGER IF EXISTS prevent_review_action_update ON review_actions")
    op.execute("DROP FUNCTION IF EXISTS prevent_review_action_mutation()")
