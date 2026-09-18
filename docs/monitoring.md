# Monitoring: watch verified numbers, alert when they move

Defining and serving a number is half the job; the other half is noticing when it moves.
A **monitor** watches a [metric](metrics.md) or a [derivation](derivations.md) over time,
learns what the value normally looks like, and raises an alert when a new value departs
from that baseline. Because a monitor only watches a *certified* target, an alert is never
about a suspect figure; it is about a **verified** one that moved. Data-observability
tools tell you a number changed; here the alert also carries the oracle's verdict, so you
know the change is real and sound to act on.

## What a monitor watches

- A **metric**: the monitor snapshots the metric's current total each check.
- A **derivation**: the monitor watches its **row count** (a volume signal) or the mean
  of a numeric column, whichever you configure.

A monitor's target must be certified: a metric (whose source is certified by
construction) or a certified derivation. You cannot monitor an unverified number.

## How anomalies are detected

On each check the monitor records a snapshot and compares the new value two ways:

- **A learned baseline.** Over the recent history (a rolling window), it estimates a
  center and a spread and flags a value that sits more than `sensitivity` spreads away.
  The default (`mad`) uses the median and a scaled median-absolute-deviation, which is
  *robust*: a past spike does not inflate the band and hide the next one. `zscore` uses
  the mean and standard deviation instead. This is the modern practice: an adaptive
  baseline that learns "normal," not a hand-set threshold.
- **Static bounds.** Optional hard `min`/`max` limits the value must stay within,
  applied even before enough history exists to learn a baseline (so a hard-limit breach
  is caught on day one).

`sensitivity` tunes the trade-off: a tighter band (e.g. `1.5`) catches small deviations
but risks noise; a wider one (`3`+) reduces false positives.

## Incidents and alerts

A run of consecutive anomalies folds into a single **incident**, so a sustained problem
raises **one** alert, not one per check. The incident closes when the value returns to
normal, which fires a recovery alert. Alerts are delivered through the configured
[webhook](models.md) (`metric.anomaly_detected` and `metric.recovered`, HMAC-signed) and
recorded in the audit log; each payload carries the value, the expected band, the reason,
and the source's oracle verdict.

## In-app notifications

Every alert also lands as an **in-app notification**: anomalies and recoveries from a
monitor, failed or slow orchestration runs, finished or failed trainings, and scheduled
model drift. The sidebar's **Inbox** shows the unread count; the Notifications page lists events newest
first, carries the oracle's verdict where the event has one, and deep-links to the
monitor, run, or model in question.

Delivery is governed per user and per event type in **Settings → Notifications**: each
type has an in-app switch and an email switch (email defaults on only for failures and
anomalies, and silently does nothing unless SMTP is configured via `SMTP_HOST`).
Disabling a type means no notification of that type is created for you at all. Old
notifications are pruned by the maintenance scheduler after `NOTIFICATION_RETENTION_DAYS`
(default 90; `0` keeps everything). The instance-wide webhook is independent of all of
this and keeps its exact payload shape.

## Running monitors

A monitor runs on a schedule (`interval_hours`) through the maintenance scheduler, and you
can run one on demand. In the web app, the **Monitors** page lists each monitor with its
latest value and status, charts its history (anomalous points highlighted), lists its
incidents, and creates monitors over your metrics and derivations. The same surface is a
`/api/monitors` REST API (create, list, check, history, delete) and two chat-agent tools
(`create_monitor`, `list_monitors`), so an agent can set up watching and report status.

The detection itself is a small, pure SDK primitive:

```python
from elbi_core import detect_anomaly

verdict = detect_anomaly([100, 101, 99, 100, 102], value=500, sensitivity=3)
verdict.anomalous  # True
verdict.reason  # "500 is … std devs above the baseline 100"
```

## Where this fits

Monitoring completes the lifecycle of a governed number: **define** it (a metric or
derivation), **verify** it (the oracle), **serve** it (dashboards, chat, the API), and now
**watch** it. Nothing else on the platform is a new kind of trust boundary; a monitor
reuses the certification you already have and adds the one thing an observability tool
cannot: an alert you can trust is about a sound number.
