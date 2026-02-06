#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# добавляем всё
git add -A

# если нет изменений — выходим спокойно
if git diff --cached --quiet; then
  echo "No changes to backup."
  exit 0
fi

# сообщение коммита
MSG="${1:-backup $(date '+%Y-%m-%d %H:%M:%S')}"

git commit -m "$MSG"
git push
echo "Backup pushed."
