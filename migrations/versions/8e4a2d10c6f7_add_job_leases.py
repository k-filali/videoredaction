"""Add durable processing job leases.

Revision ID: 8e4a2d10c6f7
Revises: 31a6b945ef72
Create Date: 2026-07-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8e4a2d10c6f7"
down_revision: str | None = "31a6b945ef72"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("processing_jobs") as batch:
        batch.add_column(sa.Column("heartbeat_at", sa.DateTime(timezone=True)))
        batch.add_column(sa.Column("lease_expires_at", sa.DateTime(timezone=True)))
        batch.create_index(
            "ix_processing_jobs_status_created",
            ["status", "created_at"],
        )
        batch.create_index(
            "ix_processing_jobs_status_lease",
            ["status", "lease_expires_at"],
        )

    op.execute(
        """
        UPDATE processing_jobs
        SET heartbeat_at = COALESCE(started_at, created_at),
            lease_expires_at = COALESCE(started_at, created_at)
        WHERE status = 'RUNNING'
        """
    )


def downgrade() -> None:
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_index("ix_processing_jobs_status_lease")
        batch.drop_index("ix_processing_jobs_status_created")
        batch.drop_column("lease_expires_at")
        batch.drop_column("heartbeat_at")
