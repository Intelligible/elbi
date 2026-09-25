A **PostHog** connector, for PostHog Cloud (US or EU) and self-hosted deployments alike:
the host is a field, because a personal API key is only ever valid against the
deployment that issued it. It syncs the project's persons, cohorts, feature flags,
insights, experiments, actions, annotations and surveys, and — only if you ask for it —
the raw event stream, read through HogQL and filtered server-side on `timestamp` so a
second sync reads only what arrived since the first. Nothing else is offered a cursor:
none of the object endpoints accepts a modified-since filter, and at a few hundred rows
apiece a full refresh costs nothing.
