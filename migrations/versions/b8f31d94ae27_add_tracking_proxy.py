"""Add a lightweight tracking proxy for edit propagation.

Revision ID: b8f31d94ae27
Revises: a1c5e8d20b39
Create Date: 2026-07-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8f31d94ae27"
down_revision: str | None = "a1c5e8d20b39"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("video_assets") as batch:
        batch.add_column(sa.Column("tracking_proxy_uri", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("video_assets") as batch:
        batch.drop_column("tracking_proxy_uri")
