# Contributing

llmkit takes changes as pull requests from a fork. Repository rules block
direct pushes and branch creation for everyone except the maintainer, so a
fork is the only way in. This page states how to get a change merged.

## Before you start

Open an issue first when the change adds a dependency, changes a public API,
alters wire behavior, or runs past a few hundred lines. A rejected issue costs
you one paragraph. A rejected pull request costs you the work.

Send these without an issue:

- A bug fix with a test that fails before the fix.
- A documentation correction.
- A new live case or hermetic fixture for behavior that already ships.

## The flow

1. Fork `dpoage/llmkit` on GitHub.
2. Clone your fork and create a branch: `git switch -c fix/thing`.
3. Commit with a sign-off: `git commit -s`. See [Sign your work](#sign-your-work).
4. Run the gate locally. See [Run the gate](#run-the-gate).
5. Push the branch to your fork.
6. Open a pull request against `master` and leave **Allow edits by maintainers**
   checked, so the maintainer can push small fixes to your branch.

## Sign your work

llmkit is AGPL-3.0, and your contribution ships under AGPL-3.0. llmkit uses the
[Developer Certificate of Origin](https://developercertificate.org/) (DCO) to
record that you have the right to send what you send. There is no separate
contributor license agreement.

Add the trailer to every commit:

```bash
git commit -s -m "provider: reject an empty tool name"
```

The trailer must carry the same email as the commit author:

```
Signed-off-by: Jane Doe <jane@example.com>
```

Git has no configuration that adds the trailer for you, so pass `-s` every
time. A hook can add it instead. Write this to `.git/hooks/prepare-commit-msg`
in your clone and run `chmod +x` on it:

```sh
#!/bin/sh
name="$(git config user.name)"
mail="$(git config user.email)"
grep -qs "^Signed-off-by: $name <$mail>$" "$1" ||
	printf '\nSigned-off-by: %s <%s>\n' "$name" "$mail" >>"$1"
```

The `DCO` check fails the pull request when a commit has no matching trailer.
Fix the last commit with `git commit --amend --signoff`. Fix a whole branch
with `git rebase --signoff master`. Both need a force-push afterward.

## Run the gate

Requires Go 1.25 or newer. Run the same commands CI runs:

```bash
go build ./...
go vet ./...
go vet -tags live ./provider/ ./agent/ ./examples/... ./decide/
go vet -tags integration ./embed/ ./sandbox/
go test -race -count=1 ./...
golangci-lint run ./...    # v2.13.2; config in .golangci.yml
gofmt -l .                 # must print nothing
```

[docs/testing.md](docs/testing.md) describes the three suites, how each one
skips, and how to run the live suite against real vendors.

One rule has teeth beyond the linters. Any change to an adapter, the agent
loop, or an `llmkit.Capabilities` field must name its hermetic test and its
live case in `provider/live_registry_test.go`. The plain `go test ./...` suite
fails when a capability has no registered live case.

## What CI runs on your pull request

Four checks gate the merge: `build / vet / gofmt / test`, `golangci-lint`,
`sandbox integration (bwrap + container CLI)`, and `DCO`.

The first workflow run on a pull request from a fork waits for the maintainer
to approve it. Checks that sit in "Expected" have not been approved yet.

The live acceptance suite does **not** run on pull requests. It holds vendor
API keys, and a pull request can edit the workflow that reads them. The
maintainer runs the live suite on `master` after the merge. If your change
touches an adapter and you hold a key for that vendor, run the lane locally
and report the result in the pull request.

## Review

The maintainer reviews every pull request and is the code owner for every
path. A merge needs one approving review from the code owner, four green
checks, and every review conversation resolved. A new push to the branch
dismisses a stale approval.

## What does not get merged

- Style-only changes: reformatting, comment rewording, renames for taste.
- A new runtime dependency. The non-test dependency set stays small.
- A new provider adapter without a recorded fixture and a live case.
- Generated or bulk-automated changes across many files, filed without an
  issue that agrees on the change first.
- A feature with no caller in the repository and no issue describing the use.

## Issues

Use the [issue forms](https://github.com/dpoage/llmkit/issues/new/choose). Blank
issues are off. A bug report needs the llmkit version, the provider, and a
repro the maintainer can run.

## Security

Do not report a vulnerability in a public issue. Read [SECURITY.md](SECURITY.md).
