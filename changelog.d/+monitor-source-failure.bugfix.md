A monitor whose source cannot be read no longer fails silently. A failed check opens an
incident and raises a `metric.source_failed` alert through the same in-app, email,
webhook and audit path an anomaly uses, once per run of failures; the next readable
value closes it as a recovery or as a new anomaly. A metric query that fails with its
own error type also no longer stops the scheduled pass over later monitors.
