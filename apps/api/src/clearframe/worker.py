import argparse
import os
from collections.abc import Sequence

import structlog

from clearframe.config import Settings, get_settings
from clearframe.database import Database
from clearframe.jobs import JobExecutionResult
from clearframe.observability import configure_observability
from clearframe.services.container import ServiceContainer


def execute_job(job_id: str, settings: Settings | None = None) -> JobExecutionResult:
    resolved_settings = settings or get_settings()
    configure_observability(resolved_settings.log_level)
    database = Database(resolved_settings.database_url)
    services = ServiceContainer.build(resolved_settings, database)
    try:
        return services.executor.execute(job_id)
    finally:
        services.runner.shutdown()
        database.engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one queued ClearFrame job.")
    parser.add_argument(
        "--job-id",
        default=os.environ.get("CLEARFRAME_JOB_ID"),
        help="Persisted processing job ID. Defaults to CLEARFRAME_JOB_ID.",
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    if not arguments.job_id:
        parser.error("--job-id or CLEARFRAME_JOB_ID is required")

    result = execute_job(arguments.job_id)
    structlog.get_logger("clearframe.worker").info(
        "worker_complete",
        job_id=arguments.job_id,
        result=result,
    )
    return 1 if result is JobExecutionResult.FAILED else 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
