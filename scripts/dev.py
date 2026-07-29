from __future__ import annotations

import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pnpm() -> str:
    executable = shutil.which("pnpm") or shutil.which("pnpm.cmd")
    if executable is None:
        raise RuntimeError("pnpm is required; run corepack enable first")
    return executable


def main() -> int:
    api = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "clearframe.main:app",
            "--app-dir",
            "apps/api/src",
            "--reload",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=ROOT,
    )
    web = subprocess.Popen(
        [_pnpm(), "--dir", "apps/web", "dev"],
        cwd=ROOT,
    )
    processes = (api, web)

    def stop(_: int | None = None, __: object | None = None) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while all(process.poll() is None for process in processes):
            for process in processes:
                try:
                    return_code = process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    continue
                return return_code
    finally:
        stop()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
