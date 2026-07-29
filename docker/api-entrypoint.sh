#!/bin/sh
set -eu
umask 027

alembic upgrade head
exec python -m clearframe.main
