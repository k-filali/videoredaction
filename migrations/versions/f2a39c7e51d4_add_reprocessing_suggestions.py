"""Add isolated context-aware reprocessing suggestions.

Revision ID: f2a39c7e51d4
Revises: b4e71cc82a16
Create Date: 2026-07-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f2a39c7e51d4"
down_revision: str | None = "b4e71cc82a16"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "reprocessing_suggestions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("video_id", sa.String(length=36), nullable=False),
        sa.Column("source_action_id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("track_id", sa.String(length=36), nullable=False),
        sa.Column("source_revision", sa.Integer(), nullable=False),
        sa.Column("class_name", sa.String(length=48), nullable=False),
        sa.Column("seed_frame_index", sa.Integer(), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=False),
        sa.Column("timestamp_ms", sa.Integer(), nullable=False),
        sa.Column("x1", sa.Float(), nullable=False),
        sa.Column("y1", sa.Float(), nullable=False),
        sa.Column("x2", sa.Float(), nullable=False),
        sa.Column("y2", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("direction", sa.String(length=16), nullable=False),
        sa.Column("propagation_method", sa.String(length=32), nullable=False),
        sa.Column("seed_locked", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["processing_jobs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_action_id"],
            ["review_actions.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["video_id"],
            ["video_assets.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_action_id",
            "frame_index",
            name="uq_reprocessing_suggestion_action_frame",
        ),
    )
    op.create_index(
        op.f("ix_reprocessing_suggestions_job_id"),
        "reprocessing_suggestions",
        ["job_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_reprocessing_suggestions_source_action_id"),
        "reprocessing_suggestions",
        ["source_action_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_reprocessing_suggestions_track_id"),
        "reprocessing_suggestions",
        ["track_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_reprocessing_suggestions_video_id"),
        "reprocessing_suggestions",
        ["video_id"],
        unique=False,
    )
    op.create_index(
        "ix_reprocessing_suggestions_video_track_frame",
        "reprocessing_suggestions",
        ["video_id", "track_id", "frame_index"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_reprocessing_suggestions_video_track_frame",
        table_name="reprocessing_suggestions",
    )
    op.drop_index(
        op.f("ix_reprocessing_suggestions_video_id"),
        table_name="reprocessing_suggestions",
    )
    op.drop_index(
        op.f("ix_reprocessing_suggestions_track_id"),
        table_name="reprocessing_suggestions",
    )
    op.drop_index(
        op.f("ix_reprocessing_suggestions_source_action_id"),
        table_name="reprocessing_suggestions",
    )
    op.drop_index(
        op.f("ix_reprocessing_suggestions_job_id"),
        table_name="reprocessing_suggestions",
    )
    op.drop_table("reprocessing_suggestions")
