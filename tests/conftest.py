"""Session-wide test setup.

Build compliance.db ONCE before any test runs. The claim-readiness tests share the
on-disk compliance store; when a COLD build (fresh/changed data -> full rebuild) lands
*inside* the first test that touches the store, it corrupts shared state for several
later tests, which then fail — but only on a cold start (they pass once the db exists).
Production never hits this: app/pipeline.py builds the store once at startup and then
serves. This fixture mirrors that "build once, then use" order so the suite is robust
to a cold compliance.db instead of depending on a pre-warmed one.
"""
import pytest


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
