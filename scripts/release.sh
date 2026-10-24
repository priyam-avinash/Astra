#!/usr/bin/env bash
# Usage: scripts/release.sh patch|minor|major "Short description of the change"
# Bumps VERSION, prepends a CHANGELOG entry, commits everything staged + the
# version files, tags vX.Y.Z and pushes branch + tag (or a release/vX.Y.Z branch
# where tag pushes are not allowed).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
part="${1:-patch}"; msg="${2:?describe the change}"
IFS=. read -r MA MI PA < VERSION
case "$part" in
  major) MA=$((MA+1)); MI=0; PA=0 ;;
  minor) MI=$((MI+1)); PA=0 ;;
  patch) PA=$((PA+1)) ;;
  *) echo "part must be patch|minor|major"; exit 1 ;;
esac
NEW="$MA.$MI.$PA"
echo "$NEW" > VERSION
today=$(date +%Y-%m-%d)
{ head -n 5 CHANGELOG.md; echo; echo "## $NEW — $today"; echo "- $msg"; tail -n +6 CHANGELOG.md; } > CHANGELOG.tmp && mv CHANGELOG.tmp CHANGELOG.md
git add VERSION CHANGELOG.md
git -c user.name="${GIT_AUTHOR_NAME:-Priyam Avinash}" -c user.email="${GIT_AUTHOR_EMAIL:-priyam.avinash@gmail.com}" \
  commit -q -m "v$NEW: $msg

${RELEASE_TRAILER:-}"
git tag -a "v$NEW" -m "v$NEW: $msg"
git push -q origin HEAD
# Version snapshot on GitHub: push the tag; if the remote refuses tags, publish a
# release/vX.Y.Z branch at the same commit instead.
if ! git push -q origin "v$NEW" 2>/dev/null; then
  git push -q origin "HEAD:refs/heads/release/v$NEW"
fi
echo "released v$NEW"
