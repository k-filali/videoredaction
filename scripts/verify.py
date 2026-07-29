from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    pnpm = shutil.which("pnpm") or shutil.which("pnpm.cmd")
    if pnpm is None:
        raise RuntimeError("pnpm is required; run corepack enable first")
    run(sys.executable, "-m", "ruff", "check", "apps/api/src", "tests", "scripts")
    run(sys.executable, "-m", "mypy", "apps/api/src", "tests", "scripts")
    run(sys.executable, "-m", "pytest")
    run(pnpm, "lint:web")
    run(pnpm, "typecheck:web")
    run(pnpm, "test:web")
    run(pnpm, "build:web")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
