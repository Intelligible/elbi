Fixed a half-written table after a warehouse sync failed part-way through. A full
refresh now keeps the previous rows until the new ones have all landed, and an
incremental sync adds its rows all at once or not at all. Fixed a source left at
`syncing` by a crash or a redeploy never syncing again: it is now marked failed about
half an hour after the run stopped, and syncs again on its usual schedule. Asking for a
sync while one is running is now refused with `409 Conflict` instead of starting a
second run beside it.
