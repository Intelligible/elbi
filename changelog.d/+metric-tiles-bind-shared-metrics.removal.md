A dashboard `metric` tile now binds a shared metric, the same definition the Metrics
page and the agent use, instead of aggregating a derivation's rows in the browser
through `viz.field` / `viz.agg`. A figure is defined once and reads the same on every
surface: the tile takes its value and format from the metric, and lineage now links
each metric to the dashboards that display it. A metric binding's filters may
reference a dashboard variable as `$name`, so a metric tile still follows the page's
filters. This is Dashboard Spec 2.0: a Spec 2.0 dashboard whose `metric` tile binds a
derivation is refused, with a message naming the fix. A dashboard in the earlier
format, whether saved by an earlier release, pushed with `elbi sync` or brought in by
`elbi import`, still opens, with each such tile shown as a note saying what it showed
and which metric to define; see `docs/upgrading.md`.
