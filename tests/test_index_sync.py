from tests.conftest import make_job


def test_sync_reports_diff_and_skips_unchanged(env, jobs):
    store, index, _ = env
    r = index.sync(jobs)
    assert (r.added, r.re_embedded, r.metadata_updated, r.unchanged) == (0, 0, 0, 4)

    changed_text = jobs[1].model_copy(update={"description": "Now building search ranking systems."})
    changed_meta = jobs[2].model_copy(update={"salary_text": "10-12 LPA"})
    new = make_job(9, title="DevOps Engineer", description="kubernetes terraform", skills=["kubernetes"])
    r = index.sync([jobs[0], changed_text, changed_meta, jobs[3], new])
    assert (r.added, r.re_embedded, r.metadata_updated, r.unchanged) == (1, 1, 1, 2)
    assert index.count() == 5
    assert store.get_jobs(["j3"])["j3"].salary_text == "10-12 LPA"


def test_prune_removes_jobs_missing_from_feed(env, jobs):
    store, index, _ = env
    r = index.sync(jobs[:2], prune=True)
    assert r.removed == 2
    assert index.count() == 2 and store.job_count() == 2


def test_remove_single(env):
    _, index, _ = env
    assert index.remove(["j1", "nope"]) == 1
    assert index.count() == 3
