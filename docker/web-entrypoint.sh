#!/bin/sh
set -eu

api_url="${CLEARFRAME_API_URL:-}"
access_token="${CLEARFRAME_ACCESS_TOKEN:-}"

case "$api_url" in
    *[!A-Za-z0-9.:/-]*)
        echo "CLEARFRAME_API_URL contains unsupported characters." >&2
        exit 1
        ;;
esac

if ! printf '%s\n' "$api_url" | grep -Eq '^https?://[A-Za-z0-9.-]+(:[0-9]+)?$'; then
    echo "CLEARFRAME_API_URL must be an HTTP(S) origin without a trailing slash." >&2
    exit 1
fi

case "$access_token" in
    "" | *[!A-Za-z0-9._~+/:=@-]*)
        echo "CLEARFRAME_ACCESS_TOKEN must be a non-empty opaque token." >&2
        exit 1
        ;;
esac
