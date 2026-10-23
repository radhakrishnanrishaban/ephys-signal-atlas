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
git ls-tree -r --name-only HEAD | grep -qE '^(internal|docs|lab)/' && die "internal files are tracked; refusing to publish"

git push -q dev HEAD:main
# --prune drops a stale public/main so a fresh (empty) public repo starts with no parent.
git fetch -q --prune public

tree="$(git rev-parse 'HEAD^{tree}')"
parent="$(git rev-parse -q --verify refs/remotes/public/main || true)"
if [[ -n "$parent" && "$(git rev-parse "$parent^{tree}")" == "$tree" ]]; then
    die "nothing changed since the last release"
fi

commit="$(git commit-tree "$tree" ${parent:+-p "$parent"} -m "$version: $message")"
git tag -a "$version" "$commit" -m "$version: $message"
git push -q public "$commit:refs/heads/main" "refs/tags/$version"
git push -q dev "refs/tags/$version"

echo "Released $version to public ($commit)."
