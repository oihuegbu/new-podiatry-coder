"""Verified-claim exemplars: few-shot retrieval from the finalized-claims
registry (tools/claims_registry.py).

The registry accumulates ground truth — claims verified either by the
consistency gate (unanimous + CLEAN) or by a human coder. Once it has real
coverage, the highest-leverage use of that truth is showing the coding model
a worked example of a similar verified encounter BEFORE it drafts a claim,
so judgment calls (E/M level, billability of borderline lines, modifier
choices) arrive anchored instead of being re-derived from scratch each run.

Modes (EXEMPLAR_MODE, default "auto"):
  off     disabled — no retrieval at all
  shadow  retrieve and RECORD what would have been injected
          (rag_context.exemplars in the result + a log line); prompts are
          untouched. This is the calibration phase: it measures neighbor
          coverage and lets retrieval quality be audited against real
          batches with zero blast radius.
  live    inject the rendered exemplar block into the coding prompts.
  auto    shadow while the registry holds <= EXEMPLAR_LIVE_THRESHOLD
          (default 500) verified claims, live once it passes that bar —
          the point where a same-scenario neighbor usually exists.

Retrieval is deterministic and dependency-free: Jaccard similarity over the
distinctive-term sets of clinical fingerprints (note category, chief
complaint, assessment, procedure list — stored per event by the registry;
older events without fingerprints fall back to their claim's own code
descriptions). Deterministic matters: exemplar choice must never become a
new source of run-to-run variance in the consistency gate.

Guardrails, regardless of mode:
  * the note's own document is never its exemplar (self-exclusion);
  * exemplars are worked examples, never lookups — the rendered block says
    so explicitly, and every downstream validator layer still runs;
  * only claims above EXEMPLAR_MIN_SIM qualify — no neighbor is injected
    just because it is the nearest of a bad lot.
"""

from __future__ import annotations

import re

from app.core.config import (
    EXEMPLAR_MODE, EXEMPLAR_LIVE_THRESHOLD, EXEMPLAR_TOP_K, EXEMPLAR_MIN_SIM)
from app.core.logger import get_logger

logger = get_logger(__name__)

# Generic clinical/coding filler that would connect any encounter to any
# other ("patient", "left", "unspecified"...). Mirrors the validator's
# evidence stopwords in spirit: linguistic filler, not medical knowledge.
_STOPWORDS = frozenset("""
    with without left right foot toe unspecified other patient history
    chronic acute status following performed procedure procedures note
    visit office established encounter initial subsequent care today
""".split())


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (text or "").lower())
            if len(w) >= 4 and w not in _STOPWORDS}


def _fingerprint_terms(event: dict) -> set[str]:
    fp = event.get("fingerprint") or {}
    parts = [fp.get("note_category", ""), fp.get("chief_complaint", ""),
             fp.get("assessment", ""), " ".join(fp.get("procedures") or [])]
    terms = _terms(" ".join(parts))
    if not terms:
        # pre-fingerprint events: the claim's own code descriptions still
        # describe the encounter clinically
        claim = event.get("claim") or {}
        descs = [e.get("description", "")
                 for arr in ("icd_codes", "cpt_codes", "hcpcs_codes")
                 for e in claim.get(arr) or []]
        terms = _terms(" ".join(descs))
    return terms


def _note_terms(note_category: str, note_sections: dict) -> set[str]:
    return _terms(" ".join([
        note_category or "",
        note_sections.get("chief_complaint", "") or "",
        note_sections.get("assessment_diagnoses", "") or "",
        note_sections.get("plan", "") or "",
    ]))


def resolve_mode(n_verified: int) -> str:
    mode = EXEMPLAR_MODE
    if mode not in ("auto", "off", "shadow", "live"):
        logger.warning(f"Unknown EXEMPLAR_MODE '{mode}' — treating as auto")
        mode = "auto"
    if mode == "auto":
        return "live" if n_verified > EXEMPLAR_LIVE_THRESHOLD else "shadow"
    return mode


def _line(entry: dict, kind: str) -> str:
    code = entry.get("code", "")
    mods = entry.get("modifiers") or []
    try:
        units = float(entry.get("units") or 1)
    except (TypeError, ValueError):
        units = 1
    desc = (entry.get("description") or "")[:80]
    bits = [code]
    if mods:
        bits.append("-" + "/-".join(str(m) for m in mods))
    if kind != "icd" and units != 1:
        bits.append(f"x{entry.get('units')}")
    if kind == "icd" and entry.get("type"):
        bits.append(f"({entry['type']})")
    return " ".join(bits) + (f" — {desc}" if desc else "")


def render_block(exemplars: list[dict]) -> str:
    """The prompt block for live mode. Explicitly framed as worked examples:
    the model must still derive every code from THIS note's documentation."""
    if not exemplars:
        return ""
    out = ["## VERIFIED SIMILAR ENCOUNTERS (worked examples — NOT lookups)",
           "These claims were verified (unanimous pipeline runs or a human "
           "coder) for clinically similar encounters. Use them ONLY to "
           "anchor coding style, level selection, and modifier usage. Every "
           "code you assign must be independently supported by THIS note's "
           "own documentation."]
    for i, ex in enumerate(exemplars, 1):
        fp = ex.get("fingerprint") or {}
        claim = ex.get("claim") or {}
        head = "; ".join(p for p in (fp.get("chief_complaint", ""),
                                     fp.get("assessment", "")) if p)
        out.append(f"### Exemplar {i} ({ex.get('verification', '?')}-verified)"
                   + (f": {head[:220]}" if head else ""))
        for arr, kind, label in (("icd_codes", "icd", "ICD-10-CM"),
                                 ("cpt_codes", "cpt", "CPT"),
                                 ("hcpcs_codes", "hcpcs", "HCPCS")):
            lines = claim.get(arr) or []
            if lines:
                out.append(f"{label}: " + " | ".join(_line(e, kind) for e in lines))
    return "\n".join(out)


def for_note(document_id: str, note_category: str, note_sections: dict,
             registry_path=None) -> tuple[str, dict]:
    """(prompt_block, info) for one note. prompt_block is '' unless mode is
    live AND qualifying neighbors exist. info always records what happened —
    in shadow mode it is the calibration measurement itself."""
    from tools.claims_registry import (
        load_events, current_view, REGISTRY_PATH)
    path = registry_path or REGISTRY_PATH
    try:
        view = current_view(load_events(path))
    except Exception as exc:
        logger.warning(f"Exemplar retrieval skipped — registry unreadable: {exc}")
        return "", {"mode": "error", "error": str(exc)}

    mode = resolve_mode(len(view))
    info: dict = {"mode": mode, "registry_size": len(view), "matches": []}
    if mode == "off":
        return "", info

    note_t = _note_terms(note_category, note_sections)
    if not note_t:
        return "", info

    scored = []
    for doc, event in view.items():
        if doc == document_id:
            continue  # a note must never be its own exemplar
        ex_t = _fingerprint_terms(event)
        if not ex_t:
            continue
        sim = len(note_t & ex_t) / len(note_t | ex_t)
        if sim >= EXEMPLAR_MIN_SIM:
            scored.append((round(sim, 4), doc, event))
    # doc id tiebreak keeps ordering deterministic across runs
    scored.sort(key=lambda t: (-t[0], t[1]))
    top = scored[:EXEMPLAR_TOP_K]

    info["matches"] = [{"document_id": d, "similarity": s,
                        "verification": e.get("verification")}
                       for s, d, e in top]
    if not top:
        return "", info

    if mode == "shadow":
        logger.info(
            f"  [EXEMPLAR shadow] {document_id}: would inject "
            f"{len(top)} exemplar(s): "
            + ", ".join(f"{d} (sim {s:.2f})" for s, d, _ in top))
        return "", info

    block = render_block([e for _, _, e in top])
    logger.info(f"  [EXEMPLAR live] {document_id}: injecting {len(top)} "
                f"exemplar(s): " + ", ".join(d for _, d, _ in top))
    return block, info
