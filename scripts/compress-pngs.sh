#!/usr/bin/env bash
set -euo pipefail

# Compress all PNG images in the repo using oxipng.
# Run manually or via the pre-push git hook.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v oxipng >/dev/null 2>&1 || {
    echo "oxipng not found. Install it first: https://github.com/shssoichiro/oxipng" >&2
    exit 1
}

echo "Compressing PNG images with oxipng..."
find . -type f -name '*.png' -not -path './.git/*' -print0 | xargs -0 -n 1 -P 4 oxipng -o 4 --strip safe
echo "Done."
