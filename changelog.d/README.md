# News fragments

Every user-facing change adds a file here. At release time `towncrier` turns every file
in this directory into a section of `CHANGELOG.md` and deletes them, so the changelog is
written by the person who made the change, in the pull request that made it.

## Do I need one?

Write a fragment if a user of the release would notice the change. Skip it otherwise,
and label the pull request `skip-news` so CI knows the omission was deliberate.

## Naming

```
+<slug>.<type>.md    (no issue number)
<issue>.<type>.md    (references that issue)
```

The leading `+` marks a fragment as having no issue number. The slug is two to four
kebab-case words naming the topic, not the branch and not the pull request:
`+example-bug-info.bugfix.md`, not `+fix-1.bugfix.md`.

If the change does have an issue, name the file after it instead, as `1.bugfix.md`,
and towncrier appends a link to the rendered entry, per `issue_format` in
`pyproject.toml`.

| Type | For |
| --- | --- |
| `feature` | A new capability a user can reach. |
| `bugfix` | Behaviour that was wrong and now is not. |
| `doc` | Documentation a user reads. |
| `removal` | Something deprecated or gone. |
| `misc` | Anything else worth announcing. |

`towncrier check` loads this directory strictly, so a misnamed file fails CI rather than
being silently ignored at release.

## Style

Write for someone reading release notes, not for someone reviewing the diff. Full
sentences, past tense, punctuation.

Say what changed, and where that is not self-explanatory, what it relieves in general
terms: the kind of situation that used to fail, not the one that happened to surface it.
Leave out the error text, the cause and the implementation.

Length follows the link. An issue-numbered fragment can be a sentence or two, because a
reader who wants the detail can follow the number. An orphan has nothing to follow, so
it carries a little more: enough for a user to tell whether the change affected them.
One or two sentences with an issue, two to four without.

No "we". No internal file or function names. Backticks around identifiers, environment
variables and anything a user types.

towncrier copies your text verbatim, line breaks included, and never reflows it. The
breaks you type are the ones that land in `CHANGELOG.md`, which keeps its lines under 90
characters, so match that. Look at the rendered result before assuming it reads well:

```bash
uv run towncrier build --draft --version <next>
```
