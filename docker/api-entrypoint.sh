#!/bin/sh
set -eu
umask 027

mkdir -p "${TMPDIR:-/tmp}"
alembic upgrade head
exec python -m clearframe.main
