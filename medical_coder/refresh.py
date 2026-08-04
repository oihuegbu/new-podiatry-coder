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

from .compiler import SourceCompiler, canonical_json


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


def discover_downloads(config: dict[str, Any]) -> tuple[list[str], str]:
    fetcher = AuthorityFetcher(set(config["allowed_hosts"]))
    page = fetcher.fetch(config["landing_url"])
    html = page.body.decode("utf-8", errors="replace")
    hrefs = re.findall(r'''href=["']([^"']+)["']''', html, flags=re.IGNORECASE)
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


class RefreshRunner:
    def __init__(self, repository_root: Path, registry_path: Path) -> None:
        self.root = repository_root.resolve()
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("schema_version") != 1:
            raise RefreshError("unsupported refresh registry schema")
        self.sources = {source["id"]: source for source in registry["sources"]}

    def refresh(self, source_id: str, pack_path: Path, output_root: Path) -> dict[str, Any]:
        if source_id not in self.sources:
            raise RefreshError(f"unknown source: {source_id}")
        config = self.sources[source_id]
        fetcher = AuthorityFetcher(set(config["allowed_hosts"]))
        if config.get("url_env"):
            url = os.environ.get(config["url_env"], "")
            if not url:
                raise RefreshError(f"required source URL environment variable is absent: {config['url_env']}")
            token = os.environ.get(config.get("token_env", "")) or None
            downloads = [fetcher.fetch(url, token)]
            effective_start = date.today().isoformat()
        else:
            urls, effective_start = discover_downloads(config)
            downloads = [fetcher.fetch(url) for url in urls]
        if config["parser"] == "delimited_archive":
            payload = parse_delimited_archives(config, downloads, effective_start)
            rows = payload
        elif config["parser"] == "verified_json":
            payload = parse_verified_json(config, downloads[0])
            rows = payload.get(config.get("container")) if config.get("container") else payload
        else:
            raise RefreshError(f"unsupported configured parser: {config['parser']}")
        if len(rows) < int(config["minimum_rows"]):
            raise RefreshError(f"parsed row count below safety floor: {len(rows)}")

        target = (self.root / config["output_path"]).resolve()
        if self.root not in target.parents:
            raise RefreshError("refresh output escapes repository")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        staging = Path(tempfile.mkdtemp(prefix=f"refresh-{source_id}-", dir="/tmp"))
        try:
            shadow = staging / "shadow"
            shutil.copytree(self.root / "data", shadow / "data")
            shutil.copytree(self.root / "source-packs", shadow / "source-packs")
            shadow_target = shadow / config["output_path"]
            shadow_target.parent.mkdir(parents=True, exist_ok=True)
            shadow_target.write_bytes(canonical_json(payload) + b"\n")
            shadow_pack = shadow / pack_path.relative_to(self.root)
            SourceCompiler(shadow).compile(shadow_pack, staging / "snapshots")

            backup = target.read_bytes() if target.exists() else None
            atomic_write(target, canonical_json(payload) + b"\n")
            try:
                final_snapshot = SourceCompiler(self.root).compile(pack_path, output_root)
                subprocess.run(
                    [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                    cwd=self.root,
                    check=True,
                )
            except Exception:
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    atomic_write(target, backup)
                raise
            provenance = {
                "source_id": source_id,
                "authority": config["authority"],
                "refreshed_at": datetime.now(timezone.utc).isoformat(),
                "effective_start": effective_start,
                "row_count": len(rows),
                "downloads": [download.__dict__ | {"body": None} for download in downloads],
                "output_path": config["output_path"],
                "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "snapshot_id": final_snapshot.name,
            }
            provenance_path = self.root / "data" / "provenance" / f"{source_id}-{run_id}.json"
            atomic_write(provenance_path, canonical_json(provenance) + b"\n")
            return provenance
        finally:
            shutil.rmtree(staging, ignore_errors=True)

