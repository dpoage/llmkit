#!/usr/bin/env bash
# Fails when a commit in <base>..<head> carries no Signed-off-by trailer whose
# email matches the commit author. Merge commits are skipped: their content
# comes from the parents, which are checked on their own.
#
#   .github/dco-check.sh master HEAD
#
# .github/workflows/dco.yml runs this on every pull request.
set -euo pipefail

base="${1:?usage: dco-check.sh <base> <head>}"
head="${2:?usage: dco-check.sh <base> <head>}"

fail=0
count=0

while IFS= read -r sha; do
	[ -n "$sha" ] || continue
	count=$((count + 1))
	author_name="$(git show -s --format='%an' "$sha")"
	author_email="$(git show -s --format='%ae' "$sha" | tr '[:upper:]' '[:lower:]')"
	signed=0

	while IFS= read -r line; do
		key="$(printf '%s' "${line%%:*}" | tr '[:upper:]' '[:lower:]')"
		[ "$key" = "signed-off-by" ] || continue
		case "$line" in
		*'<'*'>'*) ;;
		*) continue ;;
		esac
		trailer="${line#*<}"
		trailer="$(printf '%s' "${trailer%%>*}" | tr '[:upper:]' '[:lower:]')"
		if [ "$trailer" = "$author_email" ]; then
			signed=1
			break
		fi
	done < <(git show -s --format='%B' "$sha")

	if [ "$signed" -eq 1 ]; then
		printf 'ok       %s  %s\n' "${sha:0:12}" "$author_email"
		continue
	fi
	fail=1
	printf '::error::commit %s by %s <%s> carries no matching Signed-off-by trailer\n' \
		"${sha:0:12}" "$author_name" "$author_email"
done < <(git rev-list --no-merges --reverse "$base..$head")

if [ "$fail" -eq 0 ]; then
	printf '%d commit(s) signed off\n' "$count"
	exit 0
fi

cat <<EOF

Every commit needs a Developer Certificate of Origin sign-off:
https://developercertificate.org/

The trailer must carry the same email as the commit author:

    Signed-off-by: Jane Doe <jane@example.com>

Add it, then force-push your branch:

    # the last commit
    git commit --amend --signoff
    # every commit on the branch
    git rebase --signoff $base
    # every future commit in this clone
    git config format.signOff true
EOF
exit 1
