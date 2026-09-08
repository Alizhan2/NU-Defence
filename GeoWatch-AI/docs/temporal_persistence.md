# Temporal persistence

Temporal comparisons are stored in `data/runs/geowatch.sqlite3`. Schema migration 1 creates comparisons, stable change events, event review history and idempotent case links.

- A comparison ID is derived from both image hashes, dates, IoU threshold and model versions. Reopening the same inputs does not create a duplicate comparison.
- Every change gets a stable event ID and starts as `needs_review`.
- Confirm/reject operations append an immutable history row and update the current decision.
- Alerts use the comparison ID and event identity, so repeated rule evaluation is idempotent.
- The SQLite database is suitable for a single-machine MVP. Multi-user deployment requires PostgreSQL, authentication, role checks and an audit policy.
