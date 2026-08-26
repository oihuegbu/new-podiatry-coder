"""Modifier assignment — data-driven, no modifier literal in the code.

The modifier VALUES (right-side, left-side, bilateral) are DISCOVERED from
modifiers.json by matching each modifier's own description ("Right side…"), so
the engine carries no hardcoded modifier and stays valid if the modifier set
changes. The LOGIC is agnostic: a documented laterality that the chosen code's
descriptor does not already encode earns the corresponding side/bilateral
modifier.

Scope today: laterality (RT/LT) and bilateral. Distinct-service (59 / X{EPSU})
and E/M-separate (25) are the next mechanics — assigned the same way, from
NCCI signals and the E/M-with-procedure relationship — and are noted, not faked.
"""
from __future__ import annotations

from . import graph_consensus as _gc
from .models import ClinicalFact


def load_modifier_defs() -> dict:
    """{modifier_code: {description: ...}} from the authoritative modifier file.

    FAIL-CLOSED: unreadable, unparseable or empty modifier definitions raise
    `ModifierDefinitionsUnavailable`. This used to yield {} on any problem, justified by
    "the file is a REQUIRED release source, so its ABSENCE blocks certification upstream" --
    true, and exactly half the problem. A file that is PRESENT but corrupt has bytes, so the
    capability manifest sees nothing missing and certification proceeds, while every
    `_discover` below returns None and the claim goes out with no laterality, no
    distinct-service and no separately-identifiable-E/M modifier at all. Since the engine
    DISCOVERS its modifiers from these descriptions rather than naming them, an empty table
    is indistinguishable from "no modifier is warranted". (Round 5, phase 4.)
    """
    from .data_access import ModifierDefinitionsUnavailable, declared_table
    return declared_table("modifier_definitions", "modifiers",
                          ModifierDefinitionsUnavailable)


def _descr(entry) -> str:
    if isinstance(entry, dict):
        return str(entry.get("description") or entry.get("descriptor")
                   or entry.get("long_description") or "")
    return str(entry)


def _discover(defs: dict, *needles: str) -> str | None:
    """The modifier whose OWN description matches a needle — a data lookup, so
    the code below never names a modifier."""
    for code, entry in defs.items():
        low = _descr(entry).lower()
        if any(n in low for n in needles):
            return code
    return None


class ModifierEngine:
    def __init__(self, defs: dict | None = None) -> None:
        self._defs = defs if defs is not None else load_modifier_defs()
        self._right = _discover(self._defs, "right side")
        self._left = _discover(self._defs, "left side")
        self._bilateral = _discover(self._defs, "bilateral")
        # distinct-service family (X{S,E,U} preferred over 59) and E/M-separate
        self._x_structure = _discover(self._defs, "separate structure")
        self._x_encounter = _discover(self._defs, "separate encounter")
        self._x_unusual = _discover(self._defs, "non-overlapping")
        self._distinct_59 = _discover(self._defs, "distinct procedural service")
        self._em_separate = _discover(self._defs, "separately identifiable evaluation")

    def assign(self, fact: ClinicalFact, descriptor: str,
               bilat: str | None = None, reconciliation=None) -> list[str]:
        """Per-line modifiers from documented facts (laterality/bilateral), gated
        by the code's CMS bilateral indicator: '9' means the laterality concept
        does not apply to this code (e.g. a per-nail debridement) — no modifier;
        modifier 50 is applied only when the indicator is '1' (bilateral eligible).
        Empty, too, when the descriptor already encodes the side.

        issue #6 F9-R6-R2, sixth re-review: `lat` is now CLAIM-AUTHORIZED, never
        the raw attribute directly -- an unauthorized (e.g. source-negated)
        laterality must never earn a fabricated RT/LT/50 modifier."""
        lat = str(_gc.claim_authorized_value(fact, "laterality", reconciliation) or "").lower().strip()
        desc = descriptor.lower()
        # Assert a side/bilateral modifier ONLY for a code whose authoritative
        # fee-schedule record carries a bilateral-surgery indicator. A MISSING
        # indicator (None) means the code is not a laterality-bearing service -- a
        # consumed supply/implant or drug billed as a device/supply code, which is
        # not on the PFS at all -- so no RT/LT/50 (fail-closed; the A4570-RT error
        # class). '9' means the concept explicitly does not apply.
        if not bilat or bilat == "9":
            return []
        if "bilateral" in desc:
            return []
        if lat == "bilateral":
            return [self._bilateral] if (bilat == "1" and self._bilateral) else []
        if lat == "right" and "right" not in desc and self._right:
            return [self._right]
        if lat == "left" and "left" not in desc and self._left:
            return [self._left]
        return []

    def _distinct_modifier(self, a, b, reconciliation=None) -> str | None:
        """Which distinct-service modifier a pair earns — X preferred over 59,
        chosen by WHY the documentation makes the services distinct: a different
        structure/site → XS, a separate encounter → XE. Returns None when the note
        establishes NO basis for distinctness — a bypass modifier is never appended
        just to clear an NCCI edit (that is unbundling); without a documented basis
        the edit stands and the component is bundled downstream (mechanic 3).

        issue #6 F9-R6-R2, sixth re-review: laterality/anatomy are now CLAIM-
        AUTHORIZED, never the raw attribute directly -- an unauthorized value
        difference must never manufacture an NCCI-bypassing modifier from two
        services that are not actually provably distinct."""
        lat_a = _gc.claim_authorized_value(a.fact, "laterality", reconciliation)
        lat_b = _gc.claim_authorized_value(b.fact, "laterality", reconciliation)
        anat_a = _gc.claim_authorized_value(a.fact, "anatomy", reconciliation)
        anat_b = _gc.claim_authorized_value(b.fact, "anatomy", reconciliation)
        diff_site = (lat_a and lat_b and lat_a != lat_b) or \
                    (anat_a and anat_b and anat_a != anat_b)
        if diff_site and self._x_structure:
            return self._x_structure
        fa, fb = a.fact.attributes, b.fact.attributes
        diff_enc = (fa.get("encounter") and fb.get("encounter")
                    and fa.get("encounter") != fb.get("encounter"))
        if diff_enc and self._x_encounter:
            return self._x_encounter
        return None

    def assign_claim(self, result, source, reconciliation=None) -> None:
        """Claim-level modifiers that depend on relationships between lines:
          • E/M-25 when a significant, separately identifiable E/M is billed with
            a procedure on the same day;
          • a distinct-service modifier on the column-2 code of an NCCI pair that
            is bypassable-with-a-modifier (indicator '1'), which also records the
            bypass so the NCCI gate can clear it.
        """
        from .models import FactKind
        billable = result.billable_lines
        procs = [ln for ln in billable if ln.fact.kind is not FactKind.EM
                 and ln.chosen.system in ("cpt", "hcpcs")]
        ems = [ln for ln in billable if ln.fact.kind is FactKind.EM]

        # E/M-25 only when the note DOCUMENTS a significant, separately
        # identifiable E/M (not merely because a procedure was also done).
        if procs and self._em_separate:
            for ln in ems:
                if ln.fact.attributes.get("separately_identifiable") and \
                        self._em_separate not in ln.modifiers:
                    ln.modifiers.append(self._em_separate)

        for i in range(len(procs)):
            for j in range(len(procs)):
                if i == j:
                    continue
                ind = source.ncci_indicator(procs[i].chosen.code,
                                            procs[j].chosen.code, result.date_of_service)
                if ind == "1":
                    mod = self._distinct_modifier(procs[i], procs[j], reconciliation)
                    if not mod:
                        continue        # no documented basis -> do NOT bypass the edit
                    if mod not in procs[j].modifiers:
                        procs[j].modifiers.append(mod)
                    result.bypassed_ncci.append(
                        frozenset((procs[i].chosen.code, procs[j].chosen.code)))
