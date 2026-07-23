import re

OBSERVABLE_NAME = "icd_excludes1_conflict_emission"

SCHEMA_DOC = (
    "Measures unresolved ICD-10-CM Type 1 Excludes (Excludes1) conflicts "
    "that the saved record's own validator emitted: entries in the "
    "top-level 'validation_issues' list whose 'category' is "
    "'icd_excludes1_conflict'. Each firing key is "
    "'icd_excludes1_conflict|<CODE>|<CODE>' where the code pair is taken "
    "verbatim from the issue's own 'code' field (split on '|', sorted for "
    "stability), so every key ends with '|<CLAIM CODE>'. An emission "
    "change is REALIZED by a deterministic correction that removes one "
    "member of the structurally mutually exclusive pair from the record's "
    "diagnosis set (with any dependent pointer remaps), after which the "
    "validator no longer emits the conflict; nothing else can clear it. "
    "Presence = both members of an Excludes1 pair still coexist on the "
    "record; absence = the pair no longer coexists. Fails closed when the "
    "validation_issues block is missing or malformed."
)

FINDING_KINDS = ("unresolved_excludes1_conflict",)

_CATEGORY = "icd_excludes1_conflict"
_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.]{1,9}$")


def _conflict_pairs(result):
    """All Excludes1 conflict code-tuples the record's validator emits."""
    if not isinstance(result, dict):
        return []
    issues = result.get("validation_issues")
    if not isinstance(issues, list):
        return []
    pairs = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        if issue.get("category") != _CATEGORY:
            continue
        raw = issue.get("code")
        if not isinstance(raw, str) or "|" not in raw:
            continue
        codes = [c.strip() for c in raw.split("|") if c.strip()]
        if len(codes) < 2:
            continue
        if not all(_CODE_RE.match(c) for c in codes):
            continue
        pairs.append(tuple(sorted(codes)))
    return pairs


def _key(pair):
    return _CATEGORY + "|" + "|".join(pair)


def identify(result, finding):
    if not isinstance(finding, dict):
        return None, "finding is not a mapping"
    pairs = _conflict_pairs(result)
    if not pairs:
        return None, "record emits no excludes1-conflict validation issues"
    raw = finding.get("code")
    if isinstance(raw, str) and "|" in raw:
        want = tuple(sorted(c.strip() for c in raw.split("|") if c.strip()))
        matches = [p for p in pairs if p == want]
    elif isinstance(raw, str) and raw.strip():
        single = raw.strip()
        matches = [p for p in pairs if single in p]
    else:
        matches = list(pairs)
    unique = sorted(set(matches))
    if len(unique) == 1:
        return (
            _key(unique[0]),
            "resolved to the record's own excludes1-conflict emission "
            "for this code pair",
        )
    if not unique:
        return None, "no emitted excludes1 conflict matches the finding's code(s)"
    return None, "ambiguous: multiple excludes1-conflict emissions match the finding"


def signature(result):
    return {_key(p) for p in _conflict_pairs(result)}
