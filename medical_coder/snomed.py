from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator


class SnomedIntegrityError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member(archive: zipfile.ZipFile, pattern: str) -> str:
    matches = [name for name in archive.namelist() if re.search(pattern, name)]
    if len(matches) != 1:
        raise SnomedIntegrityError(
            f"RF2 archive must contain exactly one member matching {pattern!r}; found {len(matches)}"
        )
    return matches[0]


def _rows(archive: zipfile.ZipFile, member: str, required: set[str]) -> Iterator[dict[str, str]]:
    with archive.open(member) as binary:
        with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
            # RF2 is tab-delimited and does not define CSV quote semantics; a
            # literal leading quote in a clinical description must stay data.
            reader = csv.DictReader(text, delimiter="\t", quoting=csv.QUOTE_NONE)
            fields = set(reader.fieldnames or ())
            if not required.issubset(fields):
                raise SnomedIntegrityError(
                    f"RF2 member {member} lacks required columns: {sorted(required - fields)}"
                )
            for row in reader:
                yield {key: (value or "").strip() for key, value in row.items()}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def prepare_snomed_us(
    archive_path: Path,
    output_path: Path,
    *,
    minimum_concepts: int = 100_000,
    minimum_terms: int = 300_000,
    minimum_relationships: int = 500_000,
) -> dict[str, object]:
    """Compile active US Edition RF2 terminology without unpacking untrusted paths."""
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise SnomedIntegrityError("SNOMED CT archive is absent")
    source_hash = _sha256_file(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise SnomedIntegrityError(f"SNOMED CT archive CRC failed: {bad_member}")
        concept_member = _member(
            archive, r"/Snapshot/Terminology/sct2_Concept_Snapshot_[^/]+[.]txt$"
        )
        description_member = _member(
            archive, r"/Snapshot/Terminology/sct2_Description_Snapshot-en_[^/]+[.]txt$"
        )
        relationship_member = _member(
            archive, r"/Snapshot/Terminology/sct2_Relationship_Snapshot_[^/]+[.]txt$"
        )
        active_concepts: dict[str, str] = {}
        for row in _rows(
            archive, concept_member, {"id", "effectiveTime", "active"}
        ):
            if row["active"] == "1":
                active_concepts[row["id"]] = row["effectiveTime"]
        if len(active_concepts) < minimum_concepts:
            raise SnomedIntegrityError("active SNOMED CT concept count is below the safety floor")

        terms: dict[str, set[str]] = {}
        effective_times = set(active_concepts.values())
        for row in _rows(
            archive,
            description_member,
            {"effectiveTime", "active", "conceptId", "languageCode", "term"},
        ):
            if (
                row["active"] == "1"
                and row["languageCode"].casefold() == "en"
                and row["conceptId"] in active_concepts
                and row["term"]
            ):
                terms.setdefault(row["conceptId"], set()).add(row["term"])
                effective_times.add(row["effectiveTime"])
        term_count = sum(len(values) for values in terms.values())
        if term_count < minimum_terms:
            raise SnomedIntegrityError("active SNOMED CT description count is below the safety floor")

        active_relationships = sum(
            1
            for row in _rows(
                archive,
                relationship_member,
                {"active", "sourceId", "destinationId", "typeId"},
            )
            if row["active"] == "1"
            and row["sourceId"] in active_concepts
            and row["destinationId"] in active_concepts
            and row["typeId"] in active_concepts
        )
        if active_relationships < minimum_relationships:
            raise SnomedIntegrityError("active SNOMED CT relationship count is below the safety floor")

    concepts = []
    compiled_terms: dict[str, list[str]] = {}
    for concept_id in sorted(terms, key=int):
        descriptions = sorted(terms[concept_id], key=lambda value: (len(value), value.casefold(), value))
        compiled_terms[concept_id] = descriptions
        concepts.append(
            {
                "concept_id": concept_id,
                # RF2 preference is defined by the language-reference-set files,
                # which this terminology-only build does not consume.  The
                # shortest stable description is a display label, not a claim
                # about SNOMED acceptability or preference.
                "display_term": descriptions[0],
                "status": "active",
                "source_effective_time": active_concepts[concept_id],
            }
        )
    dated = sorted(value for value in effective_times if re.fullmatch(r"[0-9]{8}", value))
    release_match = re.search(r"(20[0-9]{6})T", archive_path.name)
    release_id = release_match.group(1) if release_match else (dated[-1] if dated else "unknown")
    output = {
        "metadata": {
            "authority": "NLM SNOMED CT US Edition",
            "release_id": release_id,
            "archive_name": archive_path.name,
            "archive_sha256": source_hash,
            "concept_member": concept_member,
            "description_member": description_member,
            "relationship_member": relationship_member,
            "active_concept_count": len(concepts),
            "active_description_count": term_count,
            "active_relationship_count": active_relationships,
        },
        "concepts": concepts,
        "terms": compiled_terms,
    }
    _atomic_json(output_path.resolve(), output)
    return output["metadata"]
