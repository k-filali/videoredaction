import argparse
from collections.abc import Sequence
from datetime import timedelta

import structlog

from clearframe.config import Settings, get_settings
from clearframe.database import Database
from clearframe.jobs import JobReconciler, JobReconcileSummary
from clearframe.observability import configure_observability
from clearframe.services.container import ServiceContainer


def reconcile_jobs(
    settings: Settings | None = None,
    *,
    queued_age: timedelta = timedelta(minutes=2),
) -> JobReconcileSummary:
    resolved_settings = settings or get_settings()
    configure_observability(resolved_settings.log_level)
    database = Database(resolved_settings.database_url)
    services = ServiceContainer.build(resolved_settings, database)
    try:
        return JobReconciler(
            database,
            services.runner,
            queued_age=queued_age,
        ).reconcile()
    finally:
        services.runner.shutdown()
        database.engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Redispatch queued jobs and expire abandoned workers."
    )
    parser.add_argument(
        "--queued-age-seconds",
        type=int,
        default=120,
        help="Minimum queued age before redispatch.",
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    if arguments.queued_age_seconds < 0:
        parser.error("--queued-age-seconds cannot be negative")

    summary = reconcile_jobs(
        queued_age=timedelta(seconds=arguments.queued_age_seconds)
    )
    structlog.get_logger("clearframe.reconcile").info(
        "reconcile_complete",
        queued_found=summary.queued_found,
        redispatched=summary.redispatched,
        dispatch_failed=summary.dispatch_failed,
        expired_running=summary.expired_running,
    )
    return 1 if summary.dispatch_failed else 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
