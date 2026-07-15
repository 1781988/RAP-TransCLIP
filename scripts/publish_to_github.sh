#!/usr/bin/env bash
set -euo pipefail
REMOTE_URL="${1:-https://github.com/1781988/RAP-TransCLIP.git}"
BRANCH="${2:-main}"
[[ -d .git ]] || git init
git add .
if ! git diff --cached --quiet; then git commit -m "Initialize RAP-TransCLIP research framework"; fi
git branch -M "$BRANCH"
if git remote get-url origin >/dev/null 2>&1; then git remote set-url origin "$REMOTE_URL"; else git remote add origin "$REMOTE_URL"; fi
git push -u origin "$BRANCH"
