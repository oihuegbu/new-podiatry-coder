"""Compliance-store rebuild recovery regressions."""

from app.compliance.datastore.store import ComplianceDataStore


def test_schema_rebuild_recovers_when_the_last_ingested_table_already_exists(tmp_path):
    """A crashed prior build may leave any generated table behind.

    `_create_schema` promises to reset the generated schema before recreating it;
    the final chronic-classification table must obey that lifecycle as well.
    """
    store = ComplianceDataStore(tmp_path / "compliance.db")
    store.conn.execute(
        "CREATE TABLE icd10_chronic (code TEXT NOT NULL PRIMARY KEY, chronic INTEGER NOT NULL)"
    )
    store.conn.commit()

    store._create_schema()

    columns = store.conn.execute("PRAGMA table_info(icd10_chronic)").fetchall()
    assert [column["name"] for column in columns] == ["code", "chronic"]

