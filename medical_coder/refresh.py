from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .compiler import SourceCompiler, canonical_json, sha256_file
from .mcd import parse_mcd_export
from .snomed import prepare_snomed_us


class RefreshError(RuntimeError):
    pass


def atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_copy(source: Path, target: Path) -> None:
    """Install a potentially large file without loading it into memory."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


@dataclass(frozen=True)
class Download:
    url: str
    body: bytes
    sha256: str
    etag: str | None
    last_modified: str | None


class AuthorityFetcher:
    def __init__(self, allowed_hosts: set[str], timeout: int = 300, max_bytes: int = 512 * 1024 * 1024) -> None:
        self.allowed_hosts = {host.casefold() for host in allowed_hosts}
        self.timeout = timeout
        self.max_bytes = max_bytes

    def fetch(self, url: str, token: str | None = None) -> Download:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or (parsed.hostname or "").casefold() not in self.allowed_hosts:
            raise RefreshError(f"URL is outside the configured HTTPS authority: {url}")
        headers = {"User-Agent": "gpt-medical-coder-source-compiler/0.1"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            final_url = response.geturl()
            final_host = (urllib.parse.urlparse(final_url).hostname or "").casefold()
            if final_host not in self.allowed_hosts:
                raise RefreshError(f"redirect escaped configured authority: {final_url}")
            body = response.read(self.max_bytes + 1)
            if len(body) > self.max_bytes:
                raise RefreshError("authority payload exceeds configured size ceiling")
            if not body:
                raise RefreshError("authority returned an empty payload")
            return Download(
                url=final_url,
                body=body,
                sha256=hashlib.sha256(body).hexdigest(),
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
            )


def fetch_nlm_authenticated_release(
    download_api: str,
    release_url: str,
    api_key: str,
    allowed_hosts: set[str],
    *,
    timeout: int = 600,
    max_bytes: int = 1024 * 1024 * 1024,
) -> Download:
    """Download an NLM release without persisting its query-string credential."""
    if not api_key:
        raise RefreshError("NLM release refresh requires a UMLS API key")
    release = urllib.parse.urlparse(release_url)
    endpoint = urllib.parse.urlparse(download_api)
    normalized_hosts = {host.casefold() for host in allowed_hosts}
    if (
        release.scheme != "https"
        or endpoint.scheme != "https"
        or (release.hostname or "").casefold() not in normalized_hosts
        or (endpoint.hostname or "").casefold() not in normalized_hosts
    ):
        raise RefreshError("NLM release or download endpoint is outside the configured authority")
    authenticated_url = download_api + "?" + urllib.parse.urlencode(
        {"url": release_url, "apiKey": api_key}
    )
    request = urllib.request.Request(
        authenticated_url,
        headers={"User-Agent": "gpt-medical-coder-source-compiler/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            final_host = (urllib.parse.urlparse(response.geturl()).hostname or "").casefold()
            if final_host not in normalized_hosts:
                raise RefreshError("NLM authenticated download redirected outside the configured authority")
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise RefreshError("NLM release exceeds the configured size ceiling")
            if not body:
                raise RefreshError("NLM returned an empty release")
            return Download(
                url=release_url,
                body=body,
                sha256=hashlib.sha256(body).hexdigest(),
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
            )
    except RefreshError:
        raise
    except Exception:
        raise RefreshError("authenticated NLM release download failed") from None


def parse_snomed_download(config: dict[str, Any], download: Download) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="snomed-rf2-") as directory:
        root = Path(directory)
        archive = root / Path(urllib.parse.urlparse(download.url).path).name
        atomic_write(archive, download.body)
        output = root / "snomed.json"
        prepare_snomed_us(
            archive,
            output,
            minimum_concepts=int(config["minimum_rows"]),
        )
        return json.loads(output.read_text(encoding="utf-8"))


def _normalized_identifier(value: Any) -> str:
    return "".join(character for character in str(value).upper() if character.isalnum())


def _umls_rows(archive: zipfile.ZipFile, member: str):
    # MRCONSO.RRF is a headerless, pipe-delimited UMLS contract.  Naming the
    # positions keeps the format dependency visible and rejects truncated rows.
    fields = {
        "cui": 0,
        "language": 1,
        "source": 11,
        "source_code": 13,
        "term": 14,
        "suppressed": 16,
    }
    required_width = max(fields.values()) + 1
    with archive.open(member) as binary:
        with io.TextIOWrapper(binary, encoding="utf-8", errors="strict", newline="") as text:
            for line_number, line in enumerate(text, 1):
                values = line.rstrip("\r\n").split("|")
                if len(values) < required_width:
                    raise RefreshError(f"MRCONSO row {line_number} is truncated")
                yield {name: values[index].strip() for name, index in fields.items()}


def parse_umls_mrconso_candidates(
    config: dict[str, Any],
    download: Download,
    repository_root: Path,
) -> dict[str, Any]:
    """Build recall-only SNOMED terms associated with licensed CPT concepts.

    UMLS co-concept membership is deliberately emitted as lookup terminology,
    never as an autonomous code mapping or descriptor-entailment assertion.
    """
    membership_path = (repository_root / config["membership_source"]).resolve()
    if repository_root.resolve() not in membership_path.parents or not membership_path.is_file():
        raise RefreshError("UMLS membership source is absent or escapes the repository")
    membership_document = json.loads(membership_path.read_text(encoding="utf-8"))
    membership_rows = membership_document.get("codes")
    if not isinstance(membership_rows, list):
        raise RefreshError("UMLS membership source must contain a code list")
    code_field = str(config["membership_code_field"])
    valid_codes = {
        _normalized_identifier(row.get(code_field, ""))
        for row in membership_rows
        if isinstance(row, dict) and _normalized_identifier(row.get(code_field, ""))
    }
    if not valid_codes:
        raise RefreshError("UMLS membership source contains no identifiers")
    try:
        archive = zipfile.ZipFile(io.BytesIO(download.body))
    except zipfile.BadZipFile as exc:
        raise RefreshError("UMLS payload is not the expected ZIP archive") from exc
    bad_member = archive.testzip()
    if bad_member:
        raise RefreshError(f"UMLS archive CRC failed: {bad_member}")
    members = [name for name in archive.namelist() if name.upper().endswith("/MRCONSO.RRF") or name.upper() == "MRCONSO.RRF"]
    if len(members) != 1:
        raise RefreshError(f"UMLS archive must contain exactly one MRCONSO.RRF; found {len(members)}")
    member = members[0]
    target_sources = {str(value).upper() for value in config["target_sources"]}
    clinical_sources = {str(value).upper() for value in config["clinical_sources"]}
    if not target_sources or not clinical_sources:
        raise RefreshError("UMLS candidate compilation requires explicit source vocabularies")
    cui_codes: dict[str, set[str]] = {}
    for row in _umls_rows(archive, member):
        code = _normalized_identifier(row["source_code"])
        if (
            row["language"].upper() == "ENG"
            and row["suppressed"].upper() == "N"
            and row["source"].upper() in target_sources
            and code in valid_codes
        ):
            cui_codes.setdefault(row["cui"], set()).add(code)
    terms: dict[str, set[str]] = {}
    for row in _umls_rows(archive, member):
        if (
            row["language"].upper() != "ENG"
            or row["suppressed"].upper() != "N"
            or row["source"].upper() not in clinical_sources
            or row["cui"] not in cui_codes
            or not row["term"]
        ):
            continue
        for code in cui_codes[row["cui"]]:
            terms.setdefault(code, set()).add(row["term"])
    compiled = {code: sorted(values, key=lambda value: (value.casefold(), value)) for code, values in sorted(terms.items()) if values}
    return {
        "metadata": {
            "authority": "NLM UMLS Metathesaurus",
            "release_archive": Path(urllib.parse.urlparse(download.url).path).name,
            "source_url": download.url,
            "source_sha256": download.sha256,
            "member": member,
            "target_sources": sorted(target_sources),
            "clinical_sources": sorted(clinical_sources),
            "matched_concept_count": len(cui_codes),
            "mapped_code_count": len(compiled),
            "candidate_term_count": sum(len(values) for values in compiled.values()),
            "usage": "candidate_recall_only",
        },
        "terms": compiled,
    }


def configured_hosts(config: dict[str, Any]) -> set[str]:
    hosts = {str(host).casefold().strip() for host in config.get("allowed_hosts", []) if str(host).strip()}
    host_env = config.get("allowed_hosts_env")
    if host_env:
        hosts.update(
            item.casefold().strip()
            for item in os.environ.get(host_env, "").split(",")
            if item.strip()
        )
    if not hosts or any("*" in host for host in hosts):
        raise RefreshError(f"source {config['id']} requires explicit non-wildcard authority hosts")
    return hosts


def normalize_date(value: str, fallback: str | None = None) -> str | None:
    text = (value or "").strip()
    if not text or text == "*":
        return fallback
    for pattern in (r"^(\d{4})(\d{2})(\d{2})$", r"^(\d{1,2})/(\d{1,2})/(\d{4})$"):
        match = re.match(pattern, text)
        if match:
            parts = [int(part) for part in match.groups()]
            if pattern.startswith("^(\\d{4})"):
                year, month, day = parts
            else:
                month, day, year = parts
            return date(year, month, day).isoformat()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise RefreshError(f"unrecognized source date: {text}") from exc


def release_start(match: re.Match[str]) -> str:
    values = match.groupdict()
    year = int(values["year"])
    quarter = int(values["quarter"])
    return date(year, 1 + (quarter - 1) * 3, 1).isoformat()


def _pfs_release(match: re.Match[str]) -> tuple[tuple[int, int], str]:
    year = 2000 + int(match.group("year"))
    quarter = ord(match.group("period").casefold()) - ord("a") + 1
    if quarter not in range(1, 5):
        raise RefreshError("PFS release page has an unknown period designator")
    return (year, quarter), date(year, 1 + (quarter - 1) * 3, 1).isoformat()


def discover_downloads(config: dict[str, Any]) -> tuple[list[str], str]:
    fetcher = AuthorityFetcher(configured_hosts(config))
    page = fetcher.fetch(config["landing_url"])
    html = page.body.decode("utf-8", errors="replace")
    hrefs = re.findall(r'''href=["']([^"']+)["']''', html, flags=re.IGNORECASE)
    if config.get("detail_link_regex"):
        detail_pattern = re.compile(config["detail_link_regex"])
        details = []
        for href in hrefs:
            absolute = urllib.parse.urljoin(config["landing_url"], href)
            match = detail_pattern.search(absolute)
            if match:
                key, effective = _pfs_release(match)
                details.append((key, absolute, effective))
        if not details:
            raise RefreshError(f"no release detail page matched for {config['id']}")
        newest = max(key for key, _, _ in details)
        detail_url, effective = next((url, eff) for key, url, eff in details if key == newest)
        detail = fetcher.fetch(detail_url)
        detail_hrefs = re.findall(
            r'''href=["']([^"']+)["']''', detail.body.decode("utf-8", errors="replace"),
            flags=re.IGNORECASE,
        )
        download_pattern = re.compile(config["download_link_regex"])
        selected = sorted({
            urllib.parse.urljoin(detail_url, href)
            for href in detail_hrefs
            if download_pattern.search(href)
        })
        if not selected:
            raise RefreshError(f"no release archive matched on detail page for {config['id']}")
        return selected[:1], effective
    candidates: list[tuple[tuple[int, int], str, str]] = []
    link_regex = re.compile(config["link_regex"])
    release_regex = re.compile(config["release_regex"])
    for href in hrefs:
        clean = href
        prefix = config.get("strip_prefix")
        if prefix and clean.startswith(prefix):
            clean = clean[len(prefix) :]
        absolute = urllib.parse.urljoin(config["landing_url"], clean)
        link_match = link_regex.search(absolute)
        release_match = release_regex.search(absolute)
        if not link_match or not release_match:
            continue
        values = release_match.groupdict()
        key = (int(values["year"]), int(values["quarter"]))
        candidates.append((key, absolute, release_start(release_match)))
    if not candidates:
        raise RefreshError(f"no current release links matched for {config['id']}")
    newest = max(key for key, _, _ in candidates)
    selected = sorted({url for key, url, _ in candidates if key == newest})
    if not config.get("multi_file_release"):
        selected = selected[:1]
    effective = next(effective for key, _, effective in candidates if key == newest)
    return selected, effective


def rows_from_text(text: str, aliases: dict[str, list[str]]) -> list[dict[str, str]]:
    sample = "\n".join(text.splitlines()[:80])
    delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    table = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    header_index = None
    normalized_aliases = {alias.casefold() for values in aliases.values() for alias in values}
    for index, row in enumerate(table[:100]):
        normalized_cells = [" ".join(cell.casefold().split()) for cell in row]
        matches = sum(any(alias in cell for alias in normalized_aliases) for cell in normalized_cells)
        if matches >= 2:
            header_index = index
            break
    if header_index is None:
        raise RefreshError("could not locate a header row in authority table")
    headers = [" ".join(cell.casefold().split()) for cell in table[header_index]]
    mapped: dict[str, int] = {}
    for target, values in aliases.items():
        hits = [
            index
            for index, header in enumerate(headers)
            if any(alias.casefold() in header for alias in values)
        ]
        if hits:
            mapped[target] = hits[0]
    required_identity = {"left_code", "right_code"} if "left_code" in aliases else {"target_code"}
    if not required_identity.issubset(mapped):
        raise RefreshError(f"authority header lacks identity columns: {sorted(required_identity - mapped.keys())}")
    result = []
    for source_row in table[header_index + 1 :]:
        row = {
            target: (source_row[index].strip() if index < len(source_row) else "")
            for target, index in mapped.items()
        }
        if any(row.get(field) for field in required_identity):
            result.append(row)
    return result


def parse_delimited_archives(
    config: dict[str, Any],
    downloads: list[Download],
    effective_start: str,
) -> list[dict[str, Any]]:
    identifier = re.compile(config["identifier_regex"])
    member_pattern = re.compile(config["member_regex"])
    output: list[dict[str, Any]] = []
    for download in downloads:
        try:
            archive = zipfile.ZipFile(io.BytesIO(download.body))
        except zipfile.BadZipFile as exc:
            raise RefreshError("authority payload is not the expected ZIP archive") from exc
        members = [name for name in archive.namelist() if member_pattern.search(name)]
        if not members:
            raise RefreshError("authority archive contains no configured table members")
        for member in members:
            text = archive.read(member).decode("latin-1", errors="replace")
            for row in rows_from_text(text, config["field_aliases"]):
                identity_fields = [field for field in ("left_code", "right_code", "target_code") if field in row]
                normalized = {field: row[field].replace(".", "").strip().upper() for field in identity_fields}
                if not all(identifier.fullmatch(value) for value in normalized.values()):
                    continue
                normalized.update(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in identity_fields
                    }
                )
                normalized["effective_start"] = normalize_date(row.get("effective_start", ""), effective_start)
                normalized["effective_end"] = normalize_date(row.get("effective_end", ""), None)
                normalized["source_file"] = member
                normalized["source_url"] = download.url
                normalized["source_sha256"] = download.sha256
                if normalized.get("numeric_value"):
                    try:
                        normalized["numeric_value"] = float(normalized["numeric_value"])
                    except ValueError as exc:
                        raise RefreshError("non-numeric authority limit") from exc
                output.append(normalized)
    unique = {canonical_json(row): row for row in output}
    return [unique[key] for key in sorted(unique)]


def shape_payload(config: dict[str, Any], rows: list[dict[str, Any]], downloads: list[Download]) -> Any:
    filters = config.get("row_filters", {})
    filtered = [
        row for row in rows
        if all(str(row.get(field, "")).strip() in {str(item) for item in allowed} for field, allowed in filters.items())
    ]
    if not filtered:
        raise RefreshError("all authority rows were removed by configured source filters")
    shape = config.get("output_shape", "list")
    if shape == "list":
        return filtered
    if shape != "code_map":
        raise RefreshError(f"unknown configured output shape: {shape}")
    identity = config["output_code_field"]
    field_map = config["output_field_map"]
    codes: dict[str, dict[str, Any]] = {}
    for row in filtered:
        code = str(row.get(identity, "")).strip()
        if not code:
            continue
        value = {destination: row.get(source) for destination, source in field_map.items()}
        previous = codes.get(code)
        if previous is not None and previous != value:
            raise RefreshError(f"authority output contains conflicting unqualified rows for {code}")
        codes[code] = value
    if len(codes) < int(config["minimum_rows"]):
        raise RefreshError("shaped authority output fell below the configured safety floor")
    return {
        "source": config["authority"],
        "source_urls": [download.url for download in downloads],
        "generated": datetime.now(timezone.utc).isoformat(),
        "count": len(codes),
        "codes": codes,
    }


def parse_verified_json(config: dict[str, Any], download: Download) -> Any:
    try:
        document = json.loads(download.body)
    except json.JSONDecodeError as exc:
        raise RefreshError("licensed authority response is not JSON") from exc
    rows = document.get(config.get("container")) if config.get("container") else document
    if not isinstance(rows, list):
        raise RefreshError("licensed authority JSON does not contain the configured rows")
    required = set(config["required_fields"])
    if any(not required.issubset(row) for row in rows):
        raise RefreshError("licensed authority JSON row schema changed")
    return document


def _header_index(headers: list[str], aliases: list[str]) -> int | None:
    normalized = [" ".join(value.casefold().replace("_", " ").split()) for value in headers]
    matches = [
        index for index, header in enumerate(normalized)
        if any(" ".join(alias.casefold().replace("_", " ").split()) == header for alias in aliases)
    ]
    if len(matches) > 1:
        raise RefreshError(f"licensed index header is ambiguous for aliases {aliases}")
    return matches[0] if matches else None


def parse_cpt_link_index(config: dict[str, Any], download: Download, root: Path) -> dict[str, Any]:
    text = download.body.decode(config.get("encoding", "utf-8-sig"), errors="strict")
    sample = text[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|^")
    except csv.Error as error:
        raise RefreshError("licensed CPT Link index delimiter could not be determined") from error
    table = csv.reader(io.StringIO(text), dialect)
    try:
        headers = next(table)
    except StopIteration as error:
        raise RefreshError("licensed CPT Link index is empty") from error
    main_index = _header_index(headers, config["main_term_aliases"])
    code_index = _header_index(headers, config["code_target_aliases"])
    modifier_indexes = [
        index for aliases in config["modifier_level_aliases"]
        if (index := _header_index(headers, aliases)) is not None
    ]
    see_index = _header_index(headers, config.get("see_aliases", []))
    see_also_index = _header_index(headers, config.get("see_also_aliases", []))
    if main_index is None or code_index is None or not modifier_indexes:
        raise RefreshError("licensed CPT Link index lacks the configured main-term, modifier, or code-target headers")
    cpt_document = json.loads((root / config["membership_source"]).read_text(encoding="utf-8"))
    source_rows = cpt_document.get(config.get("membership_container")) if config.get("membership_container") else cpt_document
    if not isinstance(source_rows, list):
        raise RefreshError("licensed CPT membership source changed shape")
    valid_codes = {str(row[config["membership_code_field"]]).strip() for row in source_rows}

    def cell(row: list[str], index: int | None) -> str:
        return row[index].strip() if index is not None and index < len(row) else ""

    def target_codes(value: str) -> set[str]:
        result: set[str] = set()
        for token in re.split(r"[,;\s]+", value.strip()):
            if not token:
                continue
            range_match = re.fullmatch(r"([A-Za-z0-9]+)-([A-Za-z0-9]+)", token)
            if range_match:
                left, right = range_match.groups()
                if left.isdigit() and right.isdigit() and len(left) == len(right):
                    result.update(code for code in valid_codes if left <= code <= right and code.isdigit())
                else:
                    raise RefreshError(f"licensed CPT Link index contains an unsupported range target: {token}")
            elif token in valid_codes:
                result.add(token)
        return result

    direct: dict[str, tuple[str, set[str]]] = {}
    main_targets: dict[str, set[str]] = {}
    redirects: list[tuple[str, str]] = []
    for row in table:
        parts = [cell(row, main_index), *(cell(row, index) for index in modifier_indexes)]
        parts = [part for part in parts if part]
        if not parts:
            continue
        term = " — ".join(parts)
        codes = target_codes(cell(row, code_index))
        if codes:
            normalized_term = term.casefold()
            if normalized_term not in direct:
                direct[normalized_term] = (term, set())
            direct[normalized_term][1].update(codes)
            main_targets.setdefault(parts[0].casefold(), set()).update(codes)
        for reference_index in (see_index, see_also_index):
            reference = cell(row, reference_index)
            if reference:
                redirects.append((term, reference))
    terms: dict[str, set[str]] = {code: set() for code in valid_codes}
    for canonical_term, codes in direct.values():
        for code in codes:
            terms[code].add(canonical_term)
    unresolved = []
    for term, reference in redirects:
        direct_reference = direct.get(reference.casefold())
        referenced = direct_reference[1] if direct_reference else main_targets.get(reference.casefold())
        if not referenced:
            unresolved.append(reference)
            continue
        for code in referenced:
            terms[code].add(term)
    populated = {code: sorted(values) for code, values in terms.items() if values}
    if sum(len(values) for values in populated.values()) < int(config["minimum_rows"]):
        raise RefreshError("licensed CPT Link index term count is below its configured safety floor")
    return {
        "source": "licensed AMA CPT Link index",
        "source_url": download.url,
        "source_sha256": download.sha256,
        "unresolved_cross_reference_count": len(unresolved),
        "terms": populated,
    }


class RefreshRunner:
    def __init__(self, repository_root: Path, registry_path: Path) -> None:
        self.root = repository_root.resolve()
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("schema_version") != 1:
            raise RefreshError("unsupported refresh registry schema")
        self.sources = {source["id"]: source for source in registry["sources"]}

    def _prepare(self, source_id: str) -> dict[str, Any]:
        if source_id not in self.sources:
            raise RefreshError(f"unknown source: {source_id}")
        config = self.sources[source_id]
        allowed_hosts = configured_hosts(config)
        fetcher = AuthorityFetcher(
            allowed_hosts,
            max_bytes=int(config.get("maximum_download_bytes", 512 * 1024 * 1024)),
        )
        if config.get("release_api"):
            release_document = fetcher.fetch(config["release_api"])
            try:
                releases = json.loads(release_document.body)
            except json.JSONDecodeError:
                raise RefreshError("NLM release API did not return JSON") from None
            current = [item for item in releases if item.get("current") is True and item.get("downloadUrl")]
            if len(current) != 1:
                raise RefreshError("NLM release API did not identify exactly one current release")
            api_key = os.environ.get(config["api_key_env"], "")
            archive_download = fetch_nlm_authenticated_release(
                config["download_api"],
                str(current[0]["downloadUrl"]),
                api_key,
                allowed_hosts,
                max_bytes=int(config.get("maximum_download_bytes", 1024 * 1024 * 1024)),
            )
            downloads = [release_document, archive_download]
            effective_start = str(current[0].get("releaseDate") or date.today().isoformat())
        elif config.get("url_env"):
            url = os.environ.get(config["url_env"], "")
            if not url:
                raise RefreshError(f"required source URL environment variable is absent: {config['url_env']}")
            token = os.environ.get(config.get("token_env", "")) or None
            downloads = [fetcher.fetch(url, token)]
            effective_start = date.today().isoformat()
        elif config.get("direct_url"):
            downloads = [fetcher.fetch(config["direct_url"])]
            effective_start = date.today().isoformat()
        else:
            urls, effective_start = discover_downloads(config)
            downloads = [fetcher.fetch(url) for url in urls]
        if config["parser"] == "delimited_archive":
            parsed_rows = parse_delimited_archives(config, downloads, effective_start)
            payload = shape_payload(config, parsed_rows, downloads)
            rows = payload.get("codes", {}) if isinstance(payload, dict) and "codes" in payload else payload
        elif config["parser"] == "verified_json":
            payload = parse_verified_json(config, downloads[0])
            rows = payload.get(config.get("container")) if config.get("container") else payload
        elif config["parser"] == "mcd_relational_archive":
            payload = parse_mcd_export(downloads[0].body)
            rows = payload["articles"]
        elif config["parser"] == "cpt_link_index":
            payload = parse_cpt_link_index(config, downloads[0], self.root)
            rows = payload["terms"]
        elif config["parser"] == "snomed_rf2":
            payload = parse_snomed_download(config, downloads[-1])
            rows = payload["concepts"]
        elif config["parser"] == "umls_mrconso_candidates":
            payload = parse_umls_mrconso_candidates(config, downloads[-1], self.root)
            rows = payload["terms"]
        else:
            raise RefreshError(f"unsupported configured parser: {config['parser']}")
        if len(rows) < int(config["minimum_rows"]):
            raise RefreshError(f"parsed row count below safety floor: {len(rows)}")
        target = (self.root / config["output_path"]).resolve()
        if self.root not in target.parents:
            raise RefreshError("refresh output escapes repository")
        return {
            "source_id": source_id, "config": config, "downloads": downloads,
            "effective_start": effective_start, "payload": payload, "row_count": len(rows), "target": target,
        }

    def refresh_many(self, source_ids: list[str], pack_path: Path, output_root: Path) -> list[dict[str, Any]]:
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise RefreshError("refresh source identifiers must be non-empty and unique")
        prepared = [self._prepare(source_id) for source_id in source_ids]
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        staging = Path(tempfile.mkdtemp(prefix="refresh-batch-", dir="/tmp"))
        backups: dict[Path, Path | None] = {}
        created_provenance: list[Path] = []
        try:
            shadow = staging / "shadow"
            shutil.copytree(self.root / "data", shadow / "data", copy_function=link_or_copy)
            shutil.copytree(self.root / "source-packs", shadow / "source-packs", copy_function=link_or_copy)
            for item in prepared:
                shadow_target = shadow / item["config"]["output_path"]
                atomic_write(shadow_target, canonical_json(item["payload"]) + b"\n")
            shadow_pack = shadow / pack_path.relative_to(self.root)
            SourceCompiler(shadow).compile(shadow_pack, staging / "snapshots")
            try:
                for item in prepared:
                    target = item["target"]
                    if target.exists():
                        backup = staging / "backups" / f"{len(backups):04d}-{target.name}"
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        link_or_copy(str(target), str(backup))
                        backups[target] = backup
                    else:
                        backups[target] = None
                    atomic_write(target, canonical_json(item["payload"]) + b"\n")
                final_snapshot = SourceCompiler(self.root).compile(pack_path, output_root)
                subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=self.root, check=True)
                results = []
                for item in prepared:
                    config = item["config"]
                    provenance = {
                        "source_id": item["source_id"], "authority": config["authority"],
                        "refreshed_at": datetime.now(timezone.utc).isoformat(),
                        "effective_start": item["effective_start"], "row_count": item["row_count"],
                        "downloads": [download.__dict__ | {"body": None} for download in item["downloads"]],
                        "output_path": config["output_path"],
                        "output_sha256": sha256_file(item["target"]),
                        "snapshot_id": final_snapshot.name,
                    }
                    provenance_path = self.root / "data" / "provenance" / f"{item['source_id']}-{run_id}.json"
                    atomic_write(provenance_path, canonical_json(provenance) + b"\n")
                    created_provenance.append(provenance_path)
                    results.append(provenance)
            except Exception:
                rollback_errors = []
                for provenance_path in created_provenance:
                    try:
                        provenance_path.unlink(missing_ok=True)
                    except Exception as error:
                        rollback_errors.append(f"{provenance_path}: {type(error).__name__}")
                for target, backup in backups.items():
                    try:
                        if backup is None:
                            target.unlink(missing_ok=True)
                        else:
                            atomic_copy(backup, target)
                    except Exception as error:
                        rollback_errors.append(f"{target}: {type(error).__name__}")
                if rollback_errors:
                    raise RefreshError("source refresh failed and rollback was incomplete: " + "; ".join(rollback_errors))
                raise
            return results
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def refresh(self, source_id: str, pack_path: Path, output_root: Path) -> dict[str, Any]:
        return self.refresh_many([source_id], pack_path, output_root)[0]

