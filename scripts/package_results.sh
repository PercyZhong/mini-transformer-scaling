#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

INCLUDE_BEST=false
if [[ "${1:-}" == "--include-best-checkpoints" ]]; then
  INCLUDE_BEST=true
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--include-best-checkpoints]" >&2
  exit 2
fi

if [[ ! -d outputs ]]; then
  echo "outputs/ does not exist" >&2
  exit 2
fi

mkdir -p results_packages
UTC_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
GIT_SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
ARCHIVE="results_packages/stage1-results-${UTC_STAMP}-${GIT_SHORT}.tar.gz"
MANIFEST="$(mktemp)"
trap 'rm -f "$MANIFEST"' EXIT

{
  git rev-parse HEAD 2>/dev/null || echo unknown
  git status --short 2>/dev/null || true
} > outputs/package_commit.txt

find configs outputs -type f \( \
  -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' -o \
  -name '*.png' -o -name '*.txt' -o -name '*.yaml' \
\) -print > "$MANIFEST"
if [[ "$INCLUDE_BEST" == true ]]; then
  find outputs -type f -path '*/checkpoints/best.pt' -print >> "$MANIFEST"
fi

ESTIMATED_BYTES=0
while IFS= read -r file; do
  FILE_BYTES="$(stat -c '%s' "$file")"
  ESTIMATED_BYTES=$((ESTIMATED_BYTES + FILE_BYTES))
done < "$MANIFEST"
echo "Estimated uncompressed size: $((ESTIMATED_BYTES / 1024)) KiB"
tar -czf "$ARCHIVE" -T "$MANIFEST"
(
  cd results_packages
  sha256sum "$(basename "$ARCHIVE")" > SHA256SUMS.txt
)
echo "Created $ARCHIVE and results_packages/SHA256SUMS.txt"
