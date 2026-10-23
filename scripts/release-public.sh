#!/usr/bin/env bash
set -euo pipefail

# Publish the current commit to the public repo as ONE squashed release commit + tag.
# Daily work lives in the private `dev` remote; `public` only ever sees releases.
#
# Usage: scripts/release-public.sh v0.3.0 "EXP-001 held-out Day-0 R², scoreboard v0"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

die() { echo "release-public: $*" >&2; exit 1; }

[[ $# -eq 2 ]] || die "usage: $0 <version> <message>"
version="$1"
message="$2"

[[ "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "version must look like v0.3.0"
[[ -z "$(git status --porcelain)" ]] || die "working tree not clean; commit to dev first"
git rev-parse -q --verify "refs/tags/$version" >/dev/null && die "tag $version already exists"
git ls-tree -r --name-only HEAD | grep -qE '^(internal|docs|lab)/|^CLAUDE\.md$' && die "internal files are tracked; refusing to publish"

# D-14: scoreboard rows carry private dev SHAs; the tag lists them, and the newest rows must
# come from exactly the bench and package code being published.
board="results/scoreboard.csv"
[[ -f "$board" ]] || die "$board missing"
row_commits="$(tail -n +2 "$board" | cut -d, -f2 | awk 'NF && !seen[$0]++' | paste -sd' ' -)"
newest="$(tail -n 1 "$board" | cut -d, -f2)"
git cat-file -e "$newest^{commit}" 2>/dev/null || die "scoreboard commit $newest not found"
git diff --quiet "$newest" HEAD -- bench 02_package/src \
    || die "bench/ or 02_package/src/ changed since scoreboard commit $newest; rerun python -m bench.run first"
dev_sha="$(git rev-parse --short HEAD)"
body="$version: $message

dev $dev_sha
scoreboard rows from dev commits: $row_commits (bench/ and 02_package/src/ identical to $newest)"

git push -q dev HEAD:main
# --prune drops a stale public/main so a fresh (empty) public repo starts with no parent.
git fetch -q --prune public

tree="$(git rev-parse 'HEAD^{tree}')"
parent="$(git rev-parse -q --verify refs/remotes/public/main || true)"
if [[ -n "$parent" && "$(git rev-parse "$parent^{tree}")" == "$tree" ]]; then
    die "nothing changed since the last release"
fi

commit="$(git commit-tree "$tree" ${parent:+-p "$parent"} -m "$body")"
git tag -a "$version" "$commit" -m "$body"
git push -q public "$commit:refs/heads/main" "refs/tags/$version"
git push -q dev "refs/tags/$version"

echo "Released $version to public ($commit)."
