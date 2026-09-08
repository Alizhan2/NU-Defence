from src.job_queue import JobQueue


def test_job_lifecycle_persists(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    queue = JobQueue(path)
    queued = queue.enqueue("temporal", {"zone": "a"})
    running = queue.claim_next()
    assert running.job_id == queued.job_id and running.attempts == 1
    completed = queue.complete(running.job_id, {"comparison_id": "x"})
    assert JobQueue(path).load(completed.job_id).status == "completed"


def test_dedupe_key_prevents_repeated_jobs(tmp_path):
    queue = JobQueue(tmp_path / "jobs.sqlite3")
    first = queue.enqueue("watch_zone", {"slot": 1}, dedupe_key="zone-a:slot-1")
    second = queue.enqueue("watch_zone", {"slot": 1}, dedupe_key="zone-a:slot-1")
    assert first.job_id == second.job_id
    assert len(queue.list()) == 1


def test_failure_retries_then_stops(tmp_path):
    queue = JobQueue(tmp_path / "jobs.sqlite3")
    job = queue.enqueue("temporal", {}, max_attempts=2)
    first = queue.claim_next(); assert queue.fail(first.job_id, "temporary").status == "queued"
    second = queue.claim_next(); assert queue.fail(second.job_id, "again").status == "failed"
    assert queue.claim_next() is None
