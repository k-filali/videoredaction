"""Add reviewer resolution metadata to reprocessing suggestions.

Revision ID: c7b28f914d62
Revises: f2a39c7e51d4
Create Date: 2026-07-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7b28f914d62"
down_revision: str | None = "f2a39c7e51d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("reprocessing_suggestions") as batch:
        batch.add_column(sa.Column("resolved_at", sa.DateTime(timezone=True)))
        batch.add_column(sa.Column("resolved_by_session_id", sa.String(length=64)))
        batch.add_column(sa.Column("resolution_reason_code", sa.String(length=64)))
        batch.add_column(sa.Column("resolution_action_id", sa.String(length=36)))
        batch.create_foreign_key(
            "fk_reprocessing_suggestion_resolution_action",
            "review_actions",
            ["resolution_action_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch.create_index(
            "ix_reprocessing_suggestions_resolution_action_id",
            ["resolution_action_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("reprocessing_suggestions") as batch:
        batch.drop_index("ix_reprocessing_suggestions_resolution_action_id")
        batch.drop_constraint(
            "fk_reprocessing_suggestion_resolution_action",
            type_="foreignkey",
        )
        batch.drop_column("resolution_action_id")
        batch.drop_column("resolution_reason_code")
        batch.drop_column("resolved_by_session_id")
        batch.drop_column("resolved_at")
