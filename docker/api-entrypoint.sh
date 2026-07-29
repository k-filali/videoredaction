#!/bin/sh
set -eu
umask 027

mkdir -p "${TMPDIR:-/tmp}"
if [ "${CLEARFRAME_RUN_MIGRATIONS:-false}" = "true" ]; then
    alembic upgrade head
fi
exec "$@"
