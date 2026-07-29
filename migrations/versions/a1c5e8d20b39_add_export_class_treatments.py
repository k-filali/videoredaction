"""Add per-class redaction treatments to export artifacts.

Revision ID: a1c5e8d20b39
Revises: 8e4a2d10c6f7
Create Date: 2026-07-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1c5e8d20b39"
down_revision: str | None = "8e4a2d10c6f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("export_artifacts") as batch:
        batch.add_column(
            sa.Column(
                "class_treatments",
                sa.JSON(),
                nullable=False,
                server_default="{}",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("export_artifacts") as batch:
        batch.drop_column("class_treatments")
