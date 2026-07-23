"""Measurement observables — the replay gates' extensible vocabulary.

The acceptance gates can only converge on what they can MEASURE. The
original vocabulary was one observable (the billing signature); advisory
emission was the second, added by hand when a live dispute (routine_00003:
a coverage advisory wrong for the note's documented pathway) was invisible
to billing signatures by construction. This module generalizes that
hand-built growth into infrastructure, the same way template synthesis
generalized rule-mechanic growth:

  - an OBSERVABLE is a small module that (a) resolves a reviewer's
    grounded finding to the machine identity of a phenomenon in the saved
    record and (b) computes the record's emission signature for that
    phenomenon class — deterministically, read-only, from the record
    alone;
  - built-in observables live in _BUILTINS below (reviewed, static);
  - synthesized observables (tools/observable_synthesis.py) live in
    data/rules/auto_observables/ and are re-gated on every load with the
    same whitelist AST posture as auto templates, PLUS behavioral
    meta-gates at synthesis time (determinism, record immutability,
    corpus safety, identity resolution on the triggering gap).

Contract of an observable module:

    OBSERVABLE_NAME : str    snake_case; also the registry namespace
    SCHEMA_DOC      : str    what phenomenon it measures, the key format,
                             and what surface REALIZES a change to it
    FINDING_KINDS   : tuple  reviewer finding kinds it can resolve
    def identify(result, finding) -> tuple[str | None, str]
        # (key, human description) or (None, why-not). The key MUST be
        # deterministic, resolvable from the saved record alone, and end
        # with "|<CLAIM CODE>" so the actuation machinery can scope it to
        # flip classes mechanically. Ambiguity returns None — ambiguity
        # goes to a human, never a guess.
    def signature(result) -> set[str]
        # every key of this phenomenon class currently FIRING on the
        # (fully assembled) record. Emission of key k == k in signature.
        # Must be pure: no mutation, no I/O, no randomness.

The gates then speak one language for all of them: a verified target is
(observable, key, emit); convergence is "every replay's emission matches
the adjudicated verdict with billing lines byte-identical"; inertness is
"emission unchanged on every record nobody adjudicated"; and a saved
record is rewritten (and re-reviewed) when any observable's signature
changes under a grown pack even though billing lines are identical.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import DATA_DIR  # noqa: E402

AUTO_OBSERVABLES_DIR = DATA_DIR / "rules" / "auto_observables"

_IDENTIFY_PARAMS = ("result", "finding")
_SIGNATURE_PARAMS = ("result",)


# ---------------------------------------------------------------------------
# Built-in: advisory emission (the measured, live-proven original)
# ---------------------------------------------------------------------------

def _status_str(status) -> str:
    """A scrub finding's status as the saved-file string ("WARN"), whatever
    shape the record arrived in. Records read from disk carry plain strings;
    a record assembled in memory from a bare model_dump() carries Status
    enum members, whose str() is "Status.WARN" — that coercion silently
    measured EVERY in-memory replay as advisory-free (live bug,
    routine_00003). The producers now dump mode="json" (pipeline,
    _rebuild_run), and this keeps the measurement correct even if a future
    producer forgets."""
    return str(getattr(status, "value", status) or "").upper()


def _advisory_identify(result: dict, finding: dict) -> tuple[str | None, str]:
    """Resolve an advisory_defect finding to the ONE live WARN scrub
    finding on its code. The reviewer's finding is prose; the scrub block
    is data — matching them by (code, WARN) is deterministic. Zero or
    multiple candidates are ambiguity, and ambiguity goes to a human."""
    code = str(finding.get("code") or "").upper()
    if not code:
        return None, "finding names no code"
    hits = []
    for f in ((result.get("claim_scrub") or {}).get("findings") or []):
        if not isinstance(f, dict):
            continue
        if _status_str(f.get("status")) != "WARN":
            continue
        codes = {str(c).upper() for c in (f.get("codes") or [])}
        if code in codes:
            hits.append(f)
    if not hits:
        return None, ("no live WARN scrub finding on this code — the "
                      "disputed advisory is not identifiable in the "
                      "record's scrub block")
    if len(hits) > 1:
        fids = ",".join(sorted(str(h.get("filter_id")) for h in hits))
        return None, (f"{len(hits)} WARN scrub findings on this code "
                      f"({fids}) — ambiguous which one the review disputes")
    h = hits[0]
    return (f"{h.get('filter_id')}|{code}",
            f"[{h.get('filter_id')}] {str(h.get('reason') or '')[:400]}")


def _advisory_signature(result: dict) -> set[str]:
    """Every live WARN scrub finding as 'FILTER_ID|CODE' keys."""
    out: set[str] = set()
    for f in ((result.get("claim_scrub") or {}).get("findings") or []):
        if not isinstance(f, dict):
            continue
        if _status_str(f.get("status")) != "WARN":
            continue
        fid = str(f.get("filter_id") or "")
        for c in (f.get("codes") or []):
            out.add(f"{fid}|{str(c).upper()}")
    return out


_BUILTINS: dict[str, dict] = {
    "advisory_emission": {
        "name": "advisory_emission",
        "schema_doc": (
            "Measures compliance-scrubber ADVISORIES: WARN findings in "
            "claim_scrub.findings, keyed 'FILTER_ID|CODE'. A 'must not "
            "fire' verdict is realized by a validator rule whose template "
            "calls v.suppress_scrub_advisory(filter_id, code, rule_id, "
            "authority, note) — WARN-only by contract; the scrubber "
            "records the suppression as its own PASS finding."),
        "finding_kinds": ("advisory_defect",),
        "identify": _advisory_identify,
        "signature": _advisory_signature,
        "builtin": True,
    },
}


# ---------------------------------------------------------------------------
# Static gate for synthesized observable modules
# ---------------------------------------------------------------------------

def validate_observable_source(src: str) -> list[str]:
    """Deterministic static gate for LLM-authored observable code — the
    same whitelist posture as auto templates (tiny builtins surface, no
    I/O, no dunders, no while/recursion, no literal medical codes), with
    the observable contract's exports and signatures enforced."""
    from app.validation.auto_templates import (_ALLOWED_IMPORTS,
                                               _CODE_LITERAL_RE,
                                               _FORBIDDEN_NAMES,
                                               _FORBIDDEN_NODES,
                                               _MAX_AST_NODES)
    problems: list[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        problems.append(f"module too large ({len(nodes)} AST nodes, "
                        f"cap {_MAX_AST_NODES})")

    for node in nodes:
        for cls, why in _FORBIDDEN_NODES.items():
            if isinstance(node, cls):
                problems.append(f"line {node.lineno}: {why}")
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in _ALLOWED_IMPORTS:
                    problems.append(f"line {node.lineno}: import "
                                    f"{a.name!r} not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in _ALLOWED_IMPORTS:
                problems.append(f"line {node.lineno}: from "
                                f"{node.module!r} import not allowed")
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                problems.append(f"line {node.lineno}: forbidden name "
                                f"{node.id!r}")
            if "__" in node.id:
                problems.append(f"line {node.lineno}: dunder identifier "
                                f"{node.id!r}")
        elif isinstance(node, ast.Attribute):
            if "__" in node.attr:
                problems.append(f"line {node.lineno}: dunder attribute "
                                f".{node.attr}")

    name = kinds = None
    identify_def = signature_def = None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            t = node.targets[0].id
            if t == "OBSERVABLE_NAME" and \
                    isinstance(node.value, ast.Constant) and \
                    isinstance(node.value.value, str):
                name = node.value.value
            elif t == "FINDING_KINDS":
                kinds = node.value
        elif isinstance(node, ast.FunctionDef):
            if node.name == "identify":
                identify_def = node
            elif node.name == "signature":
                signature_def = node

    if name is None:
        problems.append("missing top-level OBSERVABLE_NAME string constant")
    elif not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", name):
        problems.append(f"bad OBSERVABLE_NAME {name!r} (snake_case, "
                        f"3-41 chars)")
    elif name in _BUILTINS:
        problems.append(f"OBSERVABLE_NAME {name!r} collides with a "
                        f"built-in observable")
    if "SCHEMA_DOC" not in src:
        problems.append("missing top-level SCHEMA_DOC string constant")
    if kinds is None:
        problems.append("missing top-level FINDING_KINDS tuple")
    for fn, params, label in ((identify_def, _IDENTIFY_PARAMS, "identify"),
                              (signature_def, _SIGNATURE_PARAMS,
                               "signature")):
        if fn is None:
            problems.append(f"missing top-level def {label}(...)")
            continue
        args = tuple(a.arg for a in fn.args.args)
        if args != params:
            problems.append(f"{label}() signature must be exactly "
                            f"{params}, got {args}")
        if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
            problems.append(f"{label}() may not take *args/**kwargs/"
                            f"keyword-only args")

    # recursion cycles = unbounded execution once while loops are banned
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = {fname: {c.func.id for c in ast.walk(fn)
                     if isinstance(c, ast.Call)
                     and isinstance(c.func, ast.Name) and c.func.id in fns}
             for fname, fn in fns.items()}

    def _reaches(start: str, target: str, seen: set) -> bool:
        for callee in calls.get(start, ()):
            if callee == target or (callee not in seen
                                    and _reaches(callee, target,
                                                 seen | {callee})):
                return True
        return False

    for fname in fns:
        if _reaches(fname, fname, {fname}):
            problems.append(f"function {fname!r} participates in a "
                            f"recursion cycle — unbounded execution")
            break

    m = _CODE_LITERAL_RE.search(
        re.sub(r"\\[A-Za-z]", " ", re.sub(r"\\+\.", ".", src)))
    if m:
        problems.append(f"literal medical code {m.group()!r} in observable "
                        f"source — observables must be fully generic")
    return problems


def observable_name_of(src: str) -> str:
    """The module's OBSERVABLE_NAME constant, or ''."""
    try:
        for node in ast.parse(src).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "OBSERVABLE_NAME"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                return node.value.value
    except SyntaxError:
        pass
    return ""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, dict]] = {}


def _exec_module(src: str, path: Path) -> dict | None:
    from app.validation.auto_templates import SAFE_BUILTINS
    ns = {"re": re, "__builtins__": SAFE_BUILTINS}
    exec(compile(src, str(path), "exec"), ns)  # noqa: S102 — gated above
    name = ns.get("OBSERVABLE_NAME")
    doc = ns.get("SCHEMA_DOC")
    kinds = ns.get("FINDING_KINDS")
    ident, sig = ns.get("identify"), ns.get("signature")
    if not (isinstance(name, str) and isinstance(doc, str)
            and isinstance(kinds, (tuple, list))
            and callable(ident) and callable(sig)):
        return None
    return {"name": name, "schema_doc": doc,
            "finding_kinds": tuple(str(k) for k in kinds),
            "identify": ident, "signature": sig,
            "builtin": False, "path": str(path)}


def load_auto_observables() -> dict[str, dict]:
    """{observable_name: entry} for every synthesized observable module
    that passes the static gate and loads cleanly. Defective files are
    skipped with a warning, never fatal — a broken observable must
    degrade to 'phenomena it measured stop being measured' (its verified
    targets then simply cannot converge, which routes to a human)."""
    out: dict[str, dict] = {}
    if not AUTO_OBSERVABLES_DIR.exists():
        return out
    for f in sorted(AUTO_OBSERVABLES_DIR.glob("*.py")):
        try:
            mtime = f.stat().st_mtime
            cached = _cache.get(str(f))
            if cached and cached[0] == mtime:
                out[cached[1]["name"]] = cached[1]
                continue
            src = f.read_text(encoding="utf-8")
            problems = validate_observable_source(src)
            if problems:
                logger.warning(f"Auto observable {f.name} rejected by the "
                               f"safety gate: {problems[:3]} — skipped")
                continue
            entry = _exec_module(src, f)
            if entry is None:
                logger.warning(f"Auto observable {f.name}: missing exports "
                               f"after execution — skipped")
                continue
            _cache[str(f)] = (mtime, entry)
            out[entry["name"]] = entry
        except Exception as exc:
            logger.warning(f"Auto observable {f.name} failed to load: "
                           f"{exc!r} — skipped")
    return out


def all_observables() -> dict[str, dict]:
    """Built-ins first (reviewed, static, highest trust), then synthesized
    observables — a synthesized module can never shadow a built-in."""
    out = dict(_BUILTINS)
    for name, entry in load_auto_observables().items():
        out.setdefault(name, entry)
    return out


# ---------------------------------------------------------------------------
# Shared helpers the gates/loop/adjudicator all use
# ---------------------------------------------------------------------------

def code_of_key(key: str) -> str:
    """The claim code an observable key is scoped to — by contract the
    final '|'-separated segment."""
    return str(key).rsplit("|", 1)[-1].strip().upper()


def observable_for_finding(kind: str) -> dict | None:
    """The observable whose FINDING_KINDS covers this reviewer finding
    kind, or None (first match wins; built-ins outrank synthesized)."""
    kind = str(kind or "").lower()
    for entry in all_observables().values():
        if kind in entry["finding_kinds"]:
            return entry
    return None


def emission_of(result: dict, keys_by_observable: dict[str, set[str]]
                ) -> dict[tuple, bool]:
    """{(observable, key): fires} for the watched keys, measured on one
    fully assembled record. An observable that raises or is missing
    measures as 'unmeasurable' — every one of its keys reads False AND
    the caller-visible marker key ('<name>', '__error__') reads True, so
    a crashed measurement can never silently satisfy a 'must not fire'
    verdict (fail closed, never fail silent)."""
    obs = all_observables()
    out: dict[tuple, bool] = {}
    for name, keys in keys_by_observable.items():
        entry = obs.get(name)
        if entry is None:
            for k in keys:
                out[(name, k)] = False
            out[(name, "__error__")] = True
            continue
        try:
            sig = set(entry["signature"](result))
        except Exception as exc:
            logger.warning(f"observable {name} signature() raised {exc!r}")
            for k in keys:
                out[(name, k)] = False
            out[(name, "__error__")] = True
            continue
        for k in keys:
            out[(name, k)] = k in sig
    return out


def record_signatures(result: dict) -> dict[str, frozenset]:
    """{observable: full emission signature} of a saved record — the
    rewrite criterion for replays: any observable's signature changing
    under a grown pack (billing lines identical) means the record the
    review judged no longer exists. Observables that raise contribute a
    sentinel so a crash is visible as a change, never as 'no change'."""
    out: dict[str, frozenset] = {}
    for name, entry in all_observables().items():
        try:
            out[name] = frozenset(entry["signature"](result))
        except Exception as exc:
            logger.warning(f"observable {name} signature() raised {exc!r}")
            out[name] = frozenset({"__error__"})
    return out
