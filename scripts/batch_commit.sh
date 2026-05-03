#!/usr/bin/env bash
# batch_commit.sh — run 16 staged commits in sequence
set -e
REPO="/Users/atulanand/Desktop/AI-Fake-News-Detector"
cd "$REPO"

commit() {
  local files="$1"
  local msg="$2"
  git add $files
  git commit -m "$msg"
  echo "✅ Committed: $msg"
}

# commits are pre-staged by the Python file edits
# just execute commit calls here
