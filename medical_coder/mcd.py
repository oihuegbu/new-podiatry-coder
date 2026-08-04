from __future__ import annotations

import csv
import html
import io
import re
import zipfile
from datetime import date
from typing import Any


class McdParseError(ValueError):
    pass


_TAG = re.compile(r"<[^>]+>")
_ROLE_TAIL = re.compile(r"\b(primary|secondary)\s+diagnos\w*(?:\s+codes?)?\s*:?\s*$", re.I)
_ROLE_REFERENCE = re.compile(
    r"\brefer\s+to\s+group\s+(\d+)\s+for\s+the\s+(primary|secondary)\b[^.]{0,80}?\brequired\b",
    re.I,
)
_ROLE_CONJUNCTION = re.compile(
    r"\brequire[sd]?\s+a\s+code\s+from\s+group\s+(\d+)\b[^.]{0,120}?"
    r"\band\s+a\s+code\s+from\s+group\s+(\d+)\b",
    re.I,
)
_SCOPE_LIST = re.compile(
    r"\bcodes?\s*:?\s*\(?\s*((?:(?:[A-Z]\d{4}|\d{5})\s*(?:,\s*|and\s+|\s+)?)+)",
    re.I,
)
_SCOPE_CODE = re.compile(r"\b(?:[A-Z]\d{4}|\d{5})\b")


def paragraph_text(value: str) -> str:
    decoded = html.unescape(html.unescape(value or "")).replace("&sol;", "/")
    return " ".join(_TAG.sub(" ", decoded).split())


def _group_role(value: str) -> tuple[str, list[str]]:
    text = paragraph_text(value)
    scope = sorted({
        code
        for match in _SCOPE_LIST.finditer(text)
        for code in _SCOPE_CODE.findall(match.group(1).upper())
    })
    tail = _ROLE_TAIL.search(text)
    if not tail:
        return "unspecified", scope
    return ("primary_eligible" if tail.group(1).casefold() == "primary" else "required_secondary"), scope


def resolve_group_roles(paragraphs: dict[int, str]) -> dict[int, tuple[str, list[str]]]:
    resolved = {group: _group_role(value) for group, value in paragraphs.items()}
    explicit = {group for group, (role, _) in resolved.items() if role != "unspecified"}

    def assign(group: int, role: str) -> None:
        if group in paragraphs and group not in explicit:
            resolved[group] = (role, resolved[group][1])

    for group, value in paragraphs.items():
        text = paragraph_text(value)
        reference = _ROLE_REFERENCE.search(text)
        if reference:
            target = int(reference.group(1))
            if reference.group(2).casefold() == "secondary":
                assign(group, "primary_eligible")
                assign(target, "required_secondary")
            else:
                assign(group, "required_secondary")
                assign(target, "primary_eligible")
            continue
        conjunction = _ROLE_CONJUNCTION.search(text)
        if conjunction:
            assign(int(conjunction.group(1)), "primary_eligible")
            assign(int(conjunction.group(2)), "required_secondary")
    return resolved


def _source_date(value: str) -> str:
    text = (value or "").strip()[:10]
    for pattern in (r"^(\d{4})-(\d{2})-(\d{2})$", r"^(\d{2})/(\d{2})/(\d{4})$"):
        match = re.match(pattern, text)
        if not match:
            continue
        parts = [int(item) for item in match.groups()]
        year, month, day = parts if pattern.startswith("^(\\d{4})") else (parts[2], parts[0], parts[1])
        return date(year, month, day).isoformat()
    return ""


def _window(row: dict[str, str]) -> tuple[str, str] | None:
    status = (row.get("status") or "").strip().upper()
    if status not in {"A", "R"}:
        return None
    start = _source_date(row.get("article_eff_date") or "")
    if not start:
        return None
    if status == "A":
        return start, ""
    end = _source_date(row.get("article_end_date") or row.get("article_rev_end_date") or "")
    return (start, end) if end else None


def parse_mcd_export(raw: bytes) -> dict[str, Any]:
    try:
        outer = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as error:
        raise McdParseError("MCD response is not a ZIP archive") from error
    inner_name = next((name for name in outer.namelist() if name.casefold().endswith("_csv.zip")), None)
    try:
        archive = zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) if inner_name else outer
    except zipfile.BadZipFile as error:
        raise McdParseError("MCD inner relational archive is invalid") from error
    csv.field_size_limit(64 * 1024 * 1024)

    members = {name.casefold(): name for name in archive.namelist()}

    def rows(member: str, required: bool = True) -> list[dict[str, str]]:
        name = members.get(member.casefold())
        if not name:
            if required:
                raise McdParseError(f"MCD relational member is missing: {member}")
            return []
        value = archive.read(name).decode("latin-1", errors="replace")
        return list(csv.DictReader(io.StringIO(value)))

    article_rows = [row for row in rows("article.csv") if row.get("article_id") and _window(row)]
    if not article_rows:
        raise McdParseError("MCD export has no active or retired authoritative article versions")
    titles = {row["article_id"].strip(): (row.get("title") or "").strip() for row in article_rows}
    windows = {row["article_id"].strip(): _window(row) for row in article_rows}

    type_names = {
        (row.get("contractor_type_id") or "").strip(): next(
            (value.strip() for key, value in row.items() if key and "descr" in key.casefold() and value), ""
        )
        for row in rows("contractor_type_lookup.csv")
    }
    contractor_names: dict[str, str] = {}
    for row in rows("contractor.csv"):
        identifier = (row.get("contractor_id") or "").strip()
        name = next(
            (value.strip() for key, value in row.items() if key and ("bus_name" in key.casefold() or "name" in key.casefold()) and value),
            "",
        )
        if identifier and name:
            kind = type_names.get((row.get("contractor_type_id") or "").strip(), "")
            contractor_names[identifier] = f"{name} ({kind})" if kind else name

    state_names = {
        (row.get("state_id") or "").strip(): (row.get("state_abbrev") or "").strip().upper()
        for row in rows("state_lookup.csv")
    }
    contractor_states: dict[tuple[str, str, str], set[str]] = {}
    for row in rows("contractor_jurisdiction.csv"):
        if (row.get("term_date") or "").strip():
            continue
        key = (
            (row.get("contractor_id") or "").strip(),
            (row.get("contractor_type_id") or "").strip(),
            (row.get("contractor_version") or "").strip(),
        )
        state = state_names.get((row.get("state_id") or "").strip(), "")
        if state:
            contractor_states.setdefault(key, set()).add(state)

    article_contractors: dict[str, set[str]] = {}
    article_states: dict[str, set[str]] = {}
    for row in rows("article_x_contractor.csv"):
        policy = (row.get("article_id") or "").strip()
        contractor = (row.get("contractor_id") or "").strip()
        if policy not in titles:
            continue
        if contractor in contractor_names:
            article_contractors.setdefault(policy, set()).add(contractor_names[contractor])
        key = (
            contractor,
            (row.get("contractor_type_id") or "").strip(),
            (row.get("contractor_version") or "").strip(),
        )
        article_states.setdefault(policy, set()).update(contractor_states.get(key, set()))

    related: dict[str, set[str]] = {}
    for row in rows("article_related_documents.csv", required=False):
        policy = (row.get("article_id") or "").strip()
        related_policy = (row.get("r_lcd_id") or "").strip()
        if policy in titles and related_policy:
            related.setdefault(policy, set()).add(f"L{related_policy}")

    articles: dict[str, dict[str, Any]] = {}

    def article(policy: str) -> dict[str, Any]:
        window = windows.get(policy) or ("", "")
        return articles.setdefault(policy, {
            "policy_id": f"A{policy}",
            "title": titles.get(policy, ""),
            "contractor": " ".join(sorted(article_contractors.get(policy, set()))),
            "states": sorted(article_states.get(policy, set())),
            "related_lcds": sorted(related.get(policy, set())),
            "effective_from": window[0],
            "effective_to": window[1],
            "cpt_codes": set(),
            "covered_icd": set(),
            "noncovered_icd": set(),
        })

    for row in rows("article_x_hcpc_code.csv"):
        policy = (row.get("article_id") or "").strip()
        code = (row.get("hcpc_code_id") or "").strip().upper()
        if policy in titles and code:
            article(policy)["cpt_codes"].add(code)

    paragraphs: dict[str, dict[int, str]] = {}
    for row in rows("article_x_icd10_covered_group.csv", required=False):
        policy = (row.get("article_id") or "").strip()
        group = (row.get("icd10_covered_group") or "").strip()
        if policy in titles and group.isdigit():
            paragraphs.setdefault(policy, {})[int(group)] = row.get("paragraph") or ""
    group_metadata = {
        (policy, str(group)): {
            "group": group,
            "role": role,
            "cpt_scope": scope,
            "paragraph": paragraph_text(paragraphs[policy][group])[:400],
        }
        for policy, values in paragraphs.items()
        for group, (role, scope) in resolve_group_roles(values).items()
    }
    group_codes: dict[tuple[str, str], set[str]] = {}
    for row in rows("article_x_icd10_covered.csv"):
        policy = (row.get("article_id") or "").strip()
        diagnosis = (row.get("icd10_code_id") or "").replace(".", "").strip().upper()
        if policy not in titles or not diagnosis:
            continue
        article(policy)["covered_icd"].add(diagnosis)
        group = (row.get("icd10_covered_group") or "").strip()
        if group.isdigit():
            group_codes.setdefault((policy, group), set()).add(diagnosis)
    for (policy, group), codes in sorted(group_codes.items()):
        metadata = group_metadata.get((policy, group), {
            "group": int(group), "role": "unspecified", "cpt_scope": [], "paragraph": "",
        })
        article(policy).setdefault("covered_icd_groups", []).append({**metadata, "codes": sorted(codes)})

    for row in rows("article_x_icd10_noncovered.csv"):
        policy = (row.get("article_id") or "").strip()
        diagnosis = (row.get("icd10_code_id") or "").replace(".", "").strip().upper()
        if policy in titles and diagnosis:
            article(policy)["noncovered_icd"].add(diagnosis)

    output = []
    for value in articles.values():
        if not value["states"]:
            raise McdParseError(f"MCD article {value['policy_id']} has no authoritative jurisdiction")
        output.append({
            **value,
            "cpt_codes": sorted(value["cpt_codes"]),
            "covered_icd": sorted(value["covered_icd"]),
            "noncovered_icd": sorted(value["noncovered_icd"]),
            "covered_icd_groups": sorted(value.get("covered_icd_groups", []), key=lambda group: group["group"]),
        })
    if not output:
        raise McdParseError("MCD export produced no coverage policies")
    return {
        "source": "CMS Medicare Coverage Database relational export",
        "articles": output,
    }
