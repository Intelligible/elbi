Added a **PostHog** connector, in beta, for PostHog Cloud (US or EU) and self-hosted
deployments. It syncs a project's persons, cohorts, feature flags, insights,
experiments, actions, annotations and surveys, each as a full refresh. Raw events are
not included: read them through the Custom REST source with a HogQL query, or through
PostHog's batch export to a bucket.
