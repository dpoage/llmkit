#!/usr/bin/env bash
# Reapply the GitHub-side contribution rules: the rulesets in this directory,
# plus the repository, Actions, and security settings the fork-and-pull-request
# flow depends on. Idempotent; needs `gh` authenticated as a repository admin.
#
#   .github/rulesets/apply.sh [owner/repo]
#
# Every ruleset lists Repository admin (role id 5) with bypass_mode "always",
# so the maintainer keeps direct push, force push, and merge-without-review.
#
# require_extra_approval_for_unattributed_changes is pinned to false. GitHub
# defaults it to true, which demands a second approver whenever a commit
# author's email is not linked to a GitHub account. One maintainer cannot
# supply a second approval, so the default would block honest fork commits.
set -euo pipefail

repo="${1:-dpoage/llmkit}"
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for file in "$dir"/*.json; do
	name="$(jq -r .name "$file")"
	id="$(gh api "repos/$repo/rulesets" --jq ".[] | select(.name == \"$name\") | .id")"
	if [ -n "$id" ]; then
		gh api -X PUT "repos/$repo/rulesets/$id" --input "$file" \
			--jq '"updated  \(.name)  id=\(.id)  \(.enforcement)"'
	else
		gh api -X POST "repos/$repo/rulesets" --input "$file" \
			--jq '"created  \(.name)  id=\(.id)  \(.enforcement)"'
	fi
done

# delete_branch_on_merge      sweeps merged branches instead of leaving them.
# allow_auto_merge            queues a merge behind the required checks.
# allow_update_branch         offers contributors the "Update branch" button.
# has_wiki                    an unused, unmoderated surface.
# web_commit_signoff_required signs off commits made in the browser, so a
#                             web edit cannot fail the DCO check.
gh api -X PATCH "repos/$repo" \
	-F delete_branch_on_merge=true \
	-F allow_auto_merge=true \
	-F allow_update_branch=true \
	-F has_wiki=false \
	-F web_commit_signoff_required=true \
	--jq '"settings  delete_branch_on_merge=\(.delete_branch_on_merge)  auto_merge=\(.allow_auto_merge)  update_branch=\(.allow_update_branch)  wiki=\(.has_wiki)  web_signoff=\(.web_commit_signoff_required)"'

# Workflows on a pull request from any external contributor wait for a
# maintainer to approve the run. The default only holds back first-timers.
gh api -X PUT "repos/$repo/actions/permissions/fork-pr-contributor-approval" \
	-f approval_policy=all_external_contributors
echo "actions   fork PR approval=all_external_contributors"

# Private vulnerability reporting: a security researcher files an advisory
# instead of a public issue. SECURITY.md points at it.
gh api -X PUT "repos/$repo/private-vulnerability-reporting"
echo "security  private vulnerability reporting=enabled"

# Lever, not applied: under a spam burst, cap participation for up to six
# months. Valid limits are existing_users, contributors_only, collaborators_only.
#
#   gh api -X PUT "repos/$repo/interaction-limits" \
#       -f limit=contributors_only -f expiry=one_week

# Consequence to remember: the creation rule blocks every branch the
# maintainer does not push. Turning on Dependabot version updates later means
# adding it to that ruleset's bypass list:
#
#   {"actor_id": 29110, "actor_type": "Integration", "bypass_mode": "always"}
