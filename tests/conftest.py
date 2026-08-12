"""Session-wide test setup.

Also neutralises the deployed provenance-anchor configuration for every test — see
`_neutral_provenance_anchor_env`.

Build compliance.db ONCE before any test runs. The claim-readiness tests share the
on-disk compliance store; when a COLD build (fresh/changed data -> full rebuild) lands
*inside* the first test that touches the store, it corrupts shared state for several
later tests, which then fail — but only on a cold start (they pass once the db exists).
Production never hits this: app/pipeline.py builds the store once at startup and then
serves. This fixture mirrors that "build once, then use" order so the suite is robust
to a cold compliance.db instead of depending on a pre-warmed one.
"""
import pytest

#: The deployed runtime now REQUIRES an external terminal-head checkpoint anchor
#: (docker-compose.yml pins PROVENANCE_CHECKPOINT_REQUIRED=1; .env carries the anchor URI).
#: The canonical way to run this suite is `docker compose run app python -m pytest`, which
#: inherits exactly that configuration -- so without this fixture every test that appends to
#: a provenance store would start reaching the production anchor bucket, or fail closed
#: against an anchor it was never given. Neither is what the test is asserting.
_PROVENANCE_ANCHOR_ENV = ("PROVENANCE_CHECKPOINT_ANCHOR", "PROVENANCE_CHECKPOINT_REQUIRED",
                          "PROVENANCE_CHECKPOINT_ADOPT", "PROVENANCE_STORE_ID")


@pytest.fixture(autouse=True)
def _neutral_provenance_anchor_env(monkeypatch):
    """Every test starts from a KNOWN anchor configuration, not the deployment's.

    Tests that exercise the anchor set these themselves (monkeypatch, or an injected
    backend), so clearing them here removes an ambient dependency rather than a capability:
    a test's result must come from what the test configures, never from where it happens to
    be run. `PROVENANCE_CHECKPOINT_S3_TEST_URI` is deliberately NOT cleared -- it is the
    opt-in switch for the live-bucket module, read at collection time.
    """
    for key in _PROVENANCE_ANCHOR_ENV:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session", autouse=True)
def _prebuilt_compliance_store():
    try:
        from app.rag.code_reference import CodeReferenceDB
        from app.compliance.datastore.store import ComplianceDataStore
        CodeReferenceDB().load_all()          # creates compliance.db if absent
        ComplianceDataStore().build_or_load()  # loads the now-existing, fingerprint-matched db
    except Exception:
        # If the store genuinely cannot build, individual tests surface it themselves;
        # this fixture only removes the mid-suite cold-build ordering hazard.
        pass
    yield
