"""Declarative validator rule engine.

The deterministic layers turned out to be instances of a small number of
mechanics ("templates"): note-context gates, descriptor-tiered sibling
arbitration, Index-driven companion completion, anchored secondary demotion.
This module implements each mechanic ONCE, generically; the individual rules
live as config in data/rules/validator_rules.json (lexicons, descriptor
grammar, message text, authority citations — never medical codes).

Adding the next rule of an existing mechanic is therefore a config change a
coding-literate reviewer can author, not a new Python method: the rule pack
is versioned, diff-testable against the gold corpus, and covered by the same
regression suites as the hand-written layers it replaces.

The engine executes inside the CodingValidator and uses the validator as its
context object — its reference DB, compliance store, issue reporter, and
language helpers (negation scrub, tokenizer, stemmer) — so a migrated rule
behaves byte-for-byte like the Python method it replaces.
"""

import json
import re
from functools import lru_cache

from loguru import logger

from app.core.config import DATA_DIR

RULES_FILE = DATA_DIR / "rules" / "validator_rules.json"


class _SafeMap(dict):
    """Format map that leaves unknown placeholders literal instead of
    raising — auto-authored rules choose their own message placeholders,
    and a display template must never be able to abort rule execution."""
    def __missing__(self, key):
        return "{" + key + "}"


def _fmt(template: str, **kw) -> str:
    try:
        return str(template).format_map(_SafeMap(**kw))
    except Exception:
        return str(template)


@lru_cache(maxsize=1)
def load_rule_pack(path: str = "") -> dict:
    p = path or str(RULES_FILE)
    try:
        with open(p, encoding="utf-8") as f:
            pack = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Rule pack not found at {p} — declarative rules disabled")
        return {"version": "missing", "rules": []}
    n = sum(1 for r in pack.get("rules", []) if r.get("enabled", True))
    logger.info(f"Rule pack {pack.get('version')}: {n} active declarative rules")
    return pack


class RuleEngine:
    """Executes declarative rules against the claim arrays. `v` is the
    CodingValidator: the engine reports through v._add / the suppression
    sets so downstream behavior (result rewriting, scrubbing, consistency
    comparison) is identical to the hand-written layers."""

    def __init__(self, validator):
        self.v = validator
        self.pack = load_rule_pack()
        self.rules = {r["id"]: r for r in self.pack.get("rules", [])}

    def rule(self, rule_id: str) -> dict | None:
        r = self.rules.get(rule_id)
        return r if r and r.get("enabled", True) else None

    def run_auto_rules(self, icd, cpt, hcpcs, coding_result,
                       note_full_text: str, note_assessment_text: str):
        """Execute every enabled AUTO-GENERATED pack rule through its
        template's generic executor. Hand-written rules are invoked by id at
        their curated call sites (ordering matters for some); auto rules —
        produced by the flip-actuation pipeline (tools/auto_actuate.py) —
        dispatch here generically so accepting one is a pure data change.
        Each rule is exception-guarded: a defective auto rule must degrade
        to a skipped rule, never a failed validation."""
        arrays = {"icd_codes": icd, "cpt_codes": cpt, "hcpcs_codes": hcpcs}
        for r in self.pack.get("rules", []):
            if not r.get("auto_generated") or not r.get("enabled", True):
                continue
            rid, template = r.get("id", "?"), r.get("template", "")
            entries = arrays.get(
                (r.get("applies_to") or {}).get("array", "cpt_codes"), cpt)
            try:
                if template == "context_gate":
                    self.context_gate(rid, entries, note_full_text)
                elif template == "tiered_family_arbitration":
                    self.tiered_family_arbitration(rid, cpt, note_full_text)
                elif template == "icd_tiered_axis":
                    self.icd_tiered_axis(rid, icd, note_full_text)
                elif template == "companion_completion":
                    self.companion_completion(rid, icd)
                elif template == "residual_secondary_demotion":
                    self.residual_secondary_demotion(
                        rid, icd, coding_result, note_assessment_text)
                elif template == "documented_service_completion":
                    self.documented_service_completion(
                        rid, entries, note_full_text, icd)
                elif template == "documented_diagnosis_completion":
                    self.documented_diagnosis_completion(
                        rid, icd, coding_result, note_full_text)
                else:
                    # Self-authored templates, in descending trust order:
                    # GRADUATED (promoted into the app tree after proving
                    # themselves — static, trusted), then sandboxed
                    # candidates in data/rules/auto_templates/ (re-gated
                    # on every load).
                    from app.validation.auto_templates import \
                        load_auto_templates
                    from app.validation.graduated import GRADUATED
                    tpl = GRADUATED.get(template) or \
                        load_auto_templates().get(template)
                    if tpl is None:
                        logger.warning(f"Auto rule {rid}: unknown template "
                                       f"'{template}' — skipped")
                    elif self.rule(rid) is not None:
                        tpl["execute"](self, r, icd, cpt, hcpcs,
                                       coding_result, note_full_text,
                                       note_assessment_text)
            except Exception as exc:
                logger.warning(f"Auto rule {rid} raised {exc!r} — skipped")

    # ------------------------------------------------------------------
    # template: context_gate
    # A line whose evidence terms appear in the note ONLY inside
    # non-billable contexts (each a labeled regex, optionally conditional
    # on claim facts) is suppressed.
    # ------------------------------------------------------------------
    def context_gate(self, rule_id: str, entries, note_full_text: str):
        rule = self.rule(rule_id)
        if rule is None or not note_full_text:
            return
        v = self.v
        ap = rule["applies_to"]
        code_re = re.compile(ap["code_regex"]) if ap.get("code_regex") else None
        desc_prefix = (ap.get("descriptor_prefix") or "").lower()
        low = (note_full_text or "").lower()
        sentences = re.split(r"[.;\n]", low)
        lo, hi = rule.get("claim_surgery_range", (0, -1))
        has_surgery = any(
            (c.get("code") or "").strip().isdigit()
            and len((c.get("code") or "").strip()) == 5
            and lo <= int((c.get("code") or "").strip()) <= hi
            for c in entries)
        contexts_cfg = [(c["label"], re.compile(c["regex"]),
                         bool(c.get("requires_claim_surgery")))
                        for c in rule["contexts"]]
        act = rule["action"]
        for entry in entries:
            code = (entry.get("code") or "").strip()
            if code_re and not code_re.match(code):
                continue
            if desc_prefix:
                d = ((self.v.db.validate_cpt(code) or {})
                     .get("long_description") or "").lower()
                if not d.startswith(desc_prefix):
                    continue
            terms = self._mention_terms(rule, code)
            if not terms:
                continue
            mentions = [s for s in sentences if any(p in s for p in terms)]
            if not mentions:
                continue  # the presence gate owns the no-mention case
            labels = []
            for s in mentions:
                for label, rx, needs_surgery in contexts_cfg:
                    if needs_surgery and not has_surgery:
                        continue
                    if rx.search(s):
                        labels.append(label)
                        break
                else:
                    labels = None  # a billable mention exists
                    break
            if labels is None:
                continue
            v._non_billable_codes_to_suppress.add(code)
            v._add(
                act["severity"], code, act["category"],
                _fmt(act["message"],
                     code=code, contexts=", ".join(sorted(set(labels)))),
                act["recommendation"], denial_risk=act["denial_risk"],
            )

    def _mention_terms(self, rule, code: str) -> tuple:
        source = rule.get("mention_terms", {}).get("source", "")
        if source == "modality_descriptor_lexicon":
            return self.v._modality_synonyms(code)
        if source == "descriptor_head_stem":
            # the descriptor's own leading act noun, truncated to its stem
            # prefix so every inflection matches by substring: 'Debridement,
            # open wound...' → 'debrid' matches the note's 'debrided',
            # 'debridement', 'debriding'
            d = ((self.v.db.validate_cpt(code) or {})
                 .get("long_description") or "").lower()
            # _tokens returns a set; the LEADING word needs document order
            words = re.findall(r"[a-z]+", d)
            if not words:
                return ()
            head = self.v._stem(words[0])
            return (head[:6] if len(head) > 6 else head,)
        return ()

    # ------------------------------------------------------------------
    # template: tiered_family_arbitration
    # A code family whose descriptors encode an ordered attribute axis
    # (tissue depth, view count...): the billable member is the one the
    # note's own evidence sentences support; with no evidence at all,
    # only the lowest tier survives review.
    # ------------------------------------------------------------------
    def tiered_family_arbitration(self, rule_id: str, entries,
                                  note_full_text: str):
        rule = self.rule(rule_id)
        if rule is None or not note_full_text:
            return
        v = self.v
        fam = self._tier_family(rule_id)
        billed = [(e, fam[e.get("code", "").strip()]) for e in entries
                  if e.get("code", "").strip() in fam
                  and e.get("code", "").strip()
                  not in v._non_billable_codes_to_suppress]
        if not billed:
            return
        ev = rule["evidence"]
        if ev.get("scrub_negation", True):
            _, low_note = v._note_evidence(note_full_text)
        else:
            low_note = (note_full_text or "").lower()
        tiers = rule["tiers"]
        # A tier word owned by a DIFFERENT procedure in the same sentence is
        # not evidence of this one's level — measured live (note 009), 'wound
        # debridement with bone biopsy' upgraded the debridement to the bone
        # tier off the biopsy's own tissue word.
        other_procs = ev.get("ignore_when_followed_by", [])
        guard = (rf"(?!\s+(?:{'|'.join(map(re.escape, other_procs))})\b)"
                 if other_procs else "")
        doc_rank, doc_word = 0, ""
        stem = ev["sentence_stem"]
        for sent in re.split(r"[.;\n]", low_note):
            if stem not in sent:
                continue
            for t, r in tiers.items():
                if re.search(rf"\b{t}\b{guard}", sent) and r > doc_rank:
                    doc_rank, doc_word = r, t
        target_rank = doc_rank if doc_rank else min(r for r, _d in fam.values())
        target = next((c for c, (r, _d) in sorted(fam.items())
                       if r == target_rank), None)
        if target is None:
            return
        act = rule["action"]
        billed_codes = {e.get("code", "") for e in entries}
        for entry, (rank, desc) in billed:
            code = entry.get("code", "")
            if rank == target_rank or code == target:
                continue
            evidence = (act["evidence_found"].format(word=doc_word)
                        if doc_rank else act["evidence_missing"])
            if target in billed_codes:
                v._non_billable_codes_to_suppress.add(code)
                action_txt = act["action_dup"].format(code=code, target=target)
            else:
                entry["code"] = target
                entry["needs_review"] = True
                entry["description"] = (fam[target][1]
                                        or entry.get("description", ""))
                billed_codes.add(target)
                action_txt = act["action_swap"].format(code=code, target=target)
            v._add(
                act["severity"], code, act["category"],
                act["message"].format(code=code, desc=desc[:60],
                                      evidence=evidence, action=action_txt),
                act["recommendation"], denial_risk=act["denial_risk"],
            )

    def _tier_family(self, rule_id: str) -> dict:
        """{code: (tier rank, descriptor)} derived from the reference DB's
        own descriptors per the rule's descriptor grammar. Cached per rule."""
        cache = getattr(self, "_fam_cache", None)
        if cache is None:
            cache = self._fam_cache = {}
        if rule_id in cache:
            return cache[rule_id]
        rule = self.rules[rule_id]
        f = rule["family"]
        strip_re = re.compile(f["strip_parenthetical_regex"]) \
            if f.get("strip_parenthetical_regex") else None
        tiers = rule["tiers"]
        fam: dict = {}
        for c, i in (getattr(self.v.db, "cpt", {}) or {}).items():
            d = (i.get("long_description") or "")
            dl = d.lower()
            if not dl.startswith(f["descriptor_prefix"]):
                continue
            if any(x in dl for x in f.get("descriptor_excludes", [])):
                continue
            req = f.get("descriptor_requires_any", [])
            if req and not any(x in dl for x in req):
                continue
            own_level = strip_re.sub(" ", dl) if strip_re else dl
            ranks = [r for t, r in tiers.items() if t in own_level]
            if ranks:
                fam[c] = (max(ranks), d)
        cache[rule_id] = fam
        return fam

    # ------------------------------------------------------------------
    # template: icd_tiered_axis
    # ICD families whose FINAL character encodes an ordered severity axis
    # spelled in the descriptors' own tier words (L97-style ulcers:
    # 'limited to breakdown of skin' < 'fat layer exposed' < 'necrosis of
    # muscle' < 'necrosis of bone'). The billable member is the deepest
    # tier the note's evidence sentences document; with no tier evidence,
    # only the lowest tier is supportable. Structural mirror of the CPT
    # tiered arbitration above, over the ICD code set.
    # ------------------------------------------------------------------
    def icd_tiered_axis(self, rule_id: str, icd, note_full_text: str):
        rule = self.rule(rule_id)
        if rule is None or not note_full_text or not icd:
            return
        v = self.v
        if v.db is None:
            return
        tiers = rule["tiers"]
        req = rule["family"].get("descriptor_requires_any", [])
        ev = rule["evidence"]
        if ev.get("scrub_negation", True):
            _, low_note = v._note_evidence(note_full_text)
        else:
            low_note = (note_full_text or "").lower()
        stem = ev["sentence_stem"]
        doc_rank, doc_word = 0, ""
        for sent in re.split(r"[.;\n]", low_note):
            if stem not in sent:
                continue
            for t, r in tiers.items():
                if re.search(rf"\b{t}\b", sent) and r > doc_rank:
                    doc_rank, doc_word = r, t

        def _ranks(desc: str):
            dl = (desc or "").lower()
            if req and not any(x in dl for x in req):
                return None
            rr = [r for t, r in tiers.items() if re.search(rf"\b{t}\b", dl)]
            return max(rr) if rr else None

        act = rule["action"]
        billed_norms = {str(e.get("code") or "").replace(".", "").upper()
                        for e in icd}
        for entry in icd:
            code = (entry.get("code") or "").strip().upper()
            info = v.db.validate_icd10(code)
            if not info:
                continue
            own_rank = _ranks(info.get("description", ""))
            if own_rank is None:
                continue
            norm_c = code.replace(".", "")
            # family: siblings differing ONLY at the final character (the
            # axis position — any earlier character difference is a
            # different site/side, structurally excluded by the prefix
            # match), each carrying a tier word of its own
            fam: dict = {norm_c: (own_rank, info.get("description", ""))}
            for sib_code, sib_desc in v.db.icd10_siblings(code[:3]):
                if (sib_code == norm_c or len(sib_code) != len(norm_c)
                        or sib_code[:-1] != norm_c[:-1]):
                    continue
                r = _ranks(sib_desc)
                if r is not None:
                    fam[sib_code] = (r, sib_desc)
            if len(fam) < 2:
                continue
            avail = {r for r, _d in fam.values()}
            target_rank = (doc_rank if doc_rank in avail
                           else min(avail))
            if own_rank == target_rank:
                continue
            target = next(c for c, (r, _d) in sorted(fam.items())
                          if r == target_rank)
            dotted = (target[:3] + "." + target[3:]
                      if len(target) > 3 else target)
            if target in billed_norms:
                continue  # the supportable tier is already on the claim
            old = entry.get("code", "")
            entry["code"] = dotted
            entry["description"] = fam[target][1]
            entry["needs_review"] = True
            billed_norms.discard(norm_c)
            billed_norms.add(target)
            evidence = (act["evidence_found"].format(word=doc_word)
                        if doc_rank else act["evidence_missing"])
            v._add(
                act["severity"], old, act["category"],
                act["message"].format(code=old, target=dotted,
                                      desc=fam[target][1][:60],
                                      evidence=evidence),
                act["recommendation"].format(target=dotted),
                denial_risk=act["denial_risk"],
            )
            logger.info(f"  ICD tier arbitration: {old} → {dotted} ({evidence})")

    # ------------------------------------------------------------------
    # template: companion_completion
    # When the claim carries a carrier-condition code and a trigger code,
    # the Alphabetic Index's 'with' linkage supplies the combination code
    # the convention mandates — transposed into the billed carrier's
    # family so type variants work without being named anywhere.
    # ------------------------------------------------------------------
    def companion_completion(self, rule_id: str, icd):
        rule = self.rule(rule_id)
        v = self.v
        if rule is None or v.store is None or not icd:
            return
        trig = rule["companion_trigger"]
        trig_re = re.compile(trig["code_regex"])
        carriers, triggers = [], []
        present = {(e.get("code") or "").strip().upper().replace(".", "")
                   for e in icd if e.get("code")}
        for e in icd:
            code = (e.get("code") or "").strip().upper()
            norm = code.replace(".", "")
            if not norm:
                continue
            cat = (v.store.icd10_tabular_description(norm[:3]) or "").lower()
            own = ((v.db.validate_icd10(code) or {}).get("description")
                   or "").lower()
            if rule["carrier"]["category_desc_contains"] in cat:
                carriers.append((e, norm[:3]))
            elif trig_re.match(norm) and trig["desc_contains"] in own:
                toks = {t for t in v._tokens(own) if len(t) >= 3}
                triggers.append((e, toks))
        if not carriers or not triggers:
            return
        link = rule["index_link"]
        where = " AND ".join("term LIKE ?" for _ in link["term_contains_all"])
        rows = v.store.conn.execute(
            f"SELECT term, code FROM icd10_index_term WHERE {where}",
            [f"%{t}%" for t in link["term_contains_all"]]).fetchall()
        skip = set(link["grammar_words"]) | set(link["axis_words"])
        act, add = rule["action"], rule["add"]
        for _carrier_entry, fam3 in carriers:
            best = None  # (n extra tokens matched, code, info)
            for term, tcode in rows:
                extra = [t for t in v._tokens(term)
                         if len(t) >= 3 and t not in skip]
                if not any(extra and all(t in toks or v._stem(t) in toks
                                         for t in extra)
                           for _e, toks in triggers):
                    continue
                cand = fam3 + tcode[3:]
                info = v.db.validate_icd10(
                    cand[:3] + "." + cand[3:] if len(cand) > 3 else cand)
                if not info:
                    continue
                if best is None or len(extra) > best[0]:
                    best = (len(extra), cand, info)
            if best is None:
                continue
            _n, cand, info = best
            if any(p.startswith(cand) for p in present):
                continue  # convention already satisfied
            dotted = cand[:3] + "." + cand[3:] if len(cand) > 3 else cand
            icd.append({
                "code": dotted,
                "description": info.get("description", ""),
                "type": add["type"],
                "rationale": add["rationale"].format(family=fam3, code=dotted),
                "source_section": add["source_section"],
                "needs_review": True,
                "review_reason": add["review_reason"].format(code=dotted),
            })
            present.add(cand)
            v._add(
                act["severity"], dotted, act["category"],
                act["message"].format(
                    code=dotted, desc=info.get("description", "")[:60]),
                act["recommendation"].format(code=dotted),
                denial_risk=act["denial_risk"],
            )
            logger.info(f"  With-convention completion: added {dotted}")

    # ------------------------------------------------------------------
    # template: documented_service_completion
    # Presence arbitration for a service/supply family: when the note's
    # own sentences affirmatively document the service (its lexicon
    # stems, negation-scrubbed, outside any exclusion context), the
    # family member whose descriptor the documentation best matches
    # belongs on the claim — added if absent. When no sentence documents
    # it, billed family members are suppressed. Generalizes the
    # hand-written dispensed-footwear completion layer: candidate codes
    # come from the reference DB's descriptors per the rule's grammar,
    # never from the rule itself.
    # ------------------------------------------------------------------
    # An auto-authored family selector that matches more codes than this is
    # not a family — it's a table scan, and its "best descriptor match"
    # would be noise. Refuse to execute rather than guess.
    _SVC_FAMILY_CAP = 200

    def documented_service_completion(self, rule_id: str, entries,
                                      note_full_text: str, icd=None):
        rule = self.rule(rule_id)
        if rule is None or not note_full_text:
            return
        v = self.v
        fam = self._service_family(rule_id)
        if not fam:
            return
        if len(fam) > self._SVC_FAMILY_CAP:
            logger.warning(
                f"Rule {rule_id}: family selector matches {len(fam)} codes "
                f"(cap {self._SVC_FAMILY_CAP}) — too broad, skipped")
            return
        ev = rule["evidence"]
        if not [s for s in ev.get("service_stems", []) if str(s).strip()]:
            # No stems means NO sentence can ever document the service —
            # the suppression arm would fire unconditionally.
            logger.warning(f"Rule {rule_id}: empty service_stems — skipped")
            return
        if ev.get("scrub_negation", True):
            _, low_note = v._note_evidence(note_full_text)
        else:
            low_note = (note_full_text or "").lower()
        stems = [s.lower() for s in ev["service_stems"]]
        excl = [re.compile(c["regex"]) for c in ev.get("exclusions", [])]

        def _documented_in(low: str) -> list:
            out = []
            for sent in re.split(r"[.;\n]", low):
                if not any(s in sent for s in stems):
                    continue
                if any(rx.search(sent) for rx in excl):
                    continue
                out.append(sent)
            return out

        # Evidence asymmetry: the SUPPRESSION arm (protective — finding
        # evidence keeps a billed line) reads the full note; the ELECTION
        # arm below (claim-mutating — adds a line) reads only the clinical
        # view, where incidental tourniquet/positioning/prep anatomy is
        # removed and can never elect a code.
        documented = _documented_in(low_note)

        act = rule["action"]
        billed = {(e.get("code") or "").strip().upper() for e in entries}
        members = billed & set(fam)

        if not documented:
            for code in sorted(members):
                v._non_billable_codes_to_suppress.add(code)
                v._add(
                    act["severity"], code, act["category"],
                    _fmt(act["message_undocumented"], code=code,
                         desc=fam.get(code, "")[:60]),
                    act["recommendation"], denial_risk=act["denial_risk"],
                )
            return

        # Documented AND already represented: the service has a line on the
        # claim. Electing a different sibling here would either duplicate
        # the service or require an ordered axis this template doesn't
        # have — that arbitration belongs to tiered_family_arbitration.
        if members:
            return

        # The documented sentences elect the family member whose descriptor
        # they speak: most distinctive descriptor tokens matched wins; a tie
        # is ambiguity, and ambiguity is not evidence — do nothing.
        clin_low = low_note
        if ev.get("scrub_negation", True):
            _, clin_low = v._note_evidence(
                v._clinical_note_view(note_full_text))
        else:
            clin_low = v._clinical_note_view(note_full_text).lower()
        documented = _documented_in(clin_low)
        if not documented:
            return
        text = " ".join(documented)
        min_tokens = int(ev.get("min_descriptor_tokens", 2))
        scored = []
        for code, desc in fam.items():
            toks = {t for t in v._tokens(desc)
                    if len(t) >= 4 and t not in v._DESC_STOPWORDS}
            n = sum(1 for t in toks if t in text or v._stem(t) in text)
            if n >= min_tokens:
                scored.append((n, code))
        if not scored:
            return
        scored.sort(reverse=True)
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return
        target = scored[0][1]
        if target in billed:
            return
        # Link the added line to the diagnoses at CREATION, coverage-aware,
        # via the same selector the empty-pointer backfill uses — so a
        # rule-added code is never born with an empty box 24E (which is a
        # transmission rejection). The downstream backfill remains as the net.
        entries.append({
            "code": target,
            "description": fam[target],
            "units": 1,
            "modifiers": [],
            "linked_diagnoses": (v._select_linked_dx(target, icd)
                                 if icd else []),
            "rationale": _fmt(act["rationale_added"], code=target,
                              desc=fam[target][:60]),
            "source_section": f"validator:{rule_id}",
            "needs_review": True,
            "review_reason": _fmt(act["review_reason_added"], code=target,
                                  desc=fam[target][:60]),
        })
        v._add(
            act["severity"], target, act["category"],
            _fmt(act["message_added"], code=target, desc=fam[target][:60]),
            act["recommendation"], denial_risk=act["denial_risk"],
        )
        logger.info(f"  Service completion ({rule_id}): added {target}")

    # ------------------------------------------------------------------
    # template: documented_diagnosis_completion
    # Presence arbitration for an ICD diagnosis family — the ICD analog of
    # documented_service_completion. When the note affirmatively documents
    # the condition (evidence stems in scrubbed sentences) and no family
    # member is on the claim, the best descriptor-elected member is added;
    # when a member is billed but the condition is nowhere documented, it
    # is demoted to supporting_conditions (recorded, not billed). The
    # family comes from ICD descriptor grammar in the reference DB — never
    # a hardcoded code list.
    # ------------------------------------------------------------------
    def documented_diagnosis_completion(self, rule_id: str, icd,
                                        coding_result,
                                        note_full_text: str):
        rule = self.rule(rule_id)
        if rule is None or not note_full_text:
            return
        v = self.v
        fam = self._diagnosis_family(rule_id)
        if not fam:
            return
        if len(fam) > self._SVC_FAMILY_CAP:
            logger.warning(
                f"Rule {rule_id}: family selector matches {len(fam)} codes "
                f"(cap {self._SVC_FAMILY_CAP}) — too broad, skipped")
            return
        ev = rule["evidence"]
        stems = [s.lower() for s in ev.get("condition_stems", [])
                 if str(s).strip()]
        if not stems:
            logger.warning(f"Rule {rule_id}: empty condition_stems — skipped")
            return
        if ev.get("scrub_negation", True):
            _, low_note = v._note_evidence(note_full_text)
        else:
            low_note = (note_full_text or "").lower()
        excl = [re.compile(c["regex"]) for c in ev.get("exclusions", [])]

        def _documented_in(low: str) -> list:
            out = []
            for sent in re.split(r"[.;\n]", low):
                if not any(s in sent for s in stems):
                    continue
                if any(rx.search(sent) for rx in excl):
                    continue
                out.append(sent)
            return out

        # Evidence asymmetry (see documented_service_completion): the
        # demotion arm is protective and reads the full note; the election
        # arm below reads only the clinical view.
        documented = _documented_in(low_note)

        act = rule["action"]

        def _norm(c: str) -> str:
            return str(c or "").strip().upper().replace(".", "")

        billed = {_norm(e.get("code")): e for e in icd if e.get("code")}
        members = {c: e for c, e in billed.items() if c in fam}

        if not documented:
            # Condition nowhere documented: billed members are demoted to
            # supporting_conditions — never silently deleted. Validator-
            # added and primary entries stay: pulling a claim's primary
            # diagnosis is beyond presence arbitration.
            demoted = [e for c, e in members.items()
                       if str(e.get("type", "")).strip().lower() != "primary"
                       and not str(e.get("source_section", ""))
                       .startswith("validator:")]
            if not demoted:
                return
            icd[:] = [e for e in icd if e not in demoted]
            supporting = coding_result.setdefault(
                "supporting_conditions", [])
            for entry in demoted:
                code = entry.get("code", "")
                entry["billable_tier"] = "advisory"
                entry["needs_review"] = True
                entry["review_reason"] = _fmt(
                    act["review_reason_demoted"], code=code,
                    desc=fam.get(_norm(code), "")[:60])
                supporting.append(entry)
                v._add(
                    act["severity"], code, act["category"],
                    _fmt(act["message_undocumented"], code=code,
                         desc=fam.get(_norm(code), "")[:60]),
                    act["recommendation"], denial_risk=act["denial_risk"],
                )
                logger.info(
                    f"  Diagnosis completion ({rule_id}): demoted {code}")
            return

        # Documented AND already represented: nothing to arbitrate.
        if members:
            return

        # Elect the member whose descriptor the documented sentences speak.
        # Ties (e.g. laterality siblings both matching) are ambiguity, and
        # ambiguity is not evidence — do nothing.
        if ev.get("scrub_negation", True):
            _, clin_low = v._note_evidence(
                v._clinical_note_view(note_full_text))
        else:
            clin_low = v._clinical_note_view(note_full_text).lower()
        documented = _documented_in(clin_low)
        if not documented:
            return
        text = " ".join(documented)
        min_tokens = int(ev.get("min_descriptor_tokens", 2))
        scored = []
        for code, desc in fam.items():
            toks = {t for t in v._tokens(desc)
                    if len(t) >= 4 and t not in v._DESC_STOPWORDS}
            n = sum(1 for t in toks if t in text or v._stem(t) in text)
            if n >= min_tokens:
                scored.append((n, code))
        if not scored:
            return
        scored.sort(reverse=True)
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return
        target = scored[0][1]
        dotted = target if "." in target or len(target) <= 3 \
            else f"{target[:3]}.{target[3:]}"
        # Primary only when the claim has no primary to displace and the
        # rule explicitly claims anchor status; otherwise secondary.
        has_primary = any(
            str(e.get("type", "")).strip().lower() == "primary" for e in icd)
        add_type = ("primary"
                    if act.get("add_as_primary") and not has_primary
                    else "secondary")
        icd.append({
            "code": dotted,
            "description": fam[target],
            "type": add_type,
            "rationale": _fmt(act["rationale_added"], code=dotted,
                              desc=fam[target][:60]),
            "source_section": f"validator:{rule_id}",
            "needs_review": True,
            "review_reason": _fmt(act["review_reason_added"], code=dotted,
                                  desc=fam[target][:60]),
        })
        v._add(
            act["severity"], dotted, act["category"],
            _fmt(act["message_added"], code=dotted, desc=fam[target][:60]),
            act["recommendation"], denial_risk=act["denial_risk"],
        )
        logger.info(f"  Diagnosis completion ({rule_id}): added {dotted} "
                    f"({add_type})")

    def _diagnosis_family(self, rule_id: str) -> dict:
        """{undotted_code: descriptor} for the rule's ICD family, selected
        from the reference DB by descriptor grammar (the ICD sibling of
        _service_family). code_regex matches the DOTTED form — proposals
        cite codes the way the Tabular List prints them."""
        cache = getattr(self, "_dx_cache", None)
        if cache is None:
            cache = self._dx_cache = {}
        if rule_id in cache:
            return cache[rule_id]
        rule = self.rules[rule_id]
        f = rule["family"]
        code_re = re.compile(f["code_regex"]) if f.get("code_regex") else None
        fam: dict = {}
        for c, i in (getattr(self.v.db, "icd10", {}) or {}).items():
            dotted = c if len(c) <= 3 else f"{c[:3]}.{c[3:]}"
            if code_re and not code_re.match(dotted):
                continue
            d = i.get("description") or ""
            dl = d.lower()
            if f.get("descriptor_prefix") and \
                    not dl.startswith(f["descriptor_prefix"]):
                continue
            if any(x in dl for x in f.get("descriptor_excludes", [])):
                continue
            req = f.get("descriptor_requires_any", [])
            if req and not any(x in dl for x in req):
                continue
            fam[c] = d
        cache[rule_id] = fam
        return fam

    def _service_family(self, rule_id: str) -> dict:
        """{code: descriptor} for the rule's service family, selected from
        the reference DB by descriptor grammar. Cached per rule."""
        cache = getattr(self, "_svc_cache", None)
        if cache is None:
            cache = self._svc_cache = {}
        if rule_id in cache:
            return cache[rule_id]
        rule = self.rules[rule_id]
        f = rule["family"]
        table = (getattr(self.v.db, "hcpcs", {})
                 if (rule.get("applies_to") or {}).get("array") ==
                 "hcpcs_codes" else getattr(self.v.db, "cpt", {})) or {}
        code_re = re.compile(f["code_regex"]) if f.get("code_regex") else None
        fam: dict = {}
        for c, i in table.items():
            if code_re and not code_re.match(c):
                continue
            d = (i.get("long_description") or i.get("description") or "")
            dl = d.lower()
            if f.get("descriptor_prefix") and \
                    not dl.startswith(f["descriptor_prefix"]):
                continue
            if any(x in dl for x in f.get("descriptor_excludes", [])):
                continue
            req = f.get("descriptor_requires_any", [])
            if req and not any(x in dl for x in req):
                continue
            fam[c] = d
        cache[rule_id] = fam
        return fam

    # ------------------------------------------------------------------
    # template: residual_secondary_demotion
    # A non-primary residual-category diagnosis stays billed only when the
    # encounter's billability anchor (assessment/imaging) documents it —
    # by descriptor tokens or Index synonyms — or another billed code's
    # Tabular instruction mandates it. Otherwise it is demoted to
    # supporting_conditions: recorded, not billed.
    # ------------------------------------------------------------------
    def residual_secondary_demotion(self, rule_id: str, icd, coding_result,
                                    note_assessment_text: str):
        rule = self.rule(rule_id)
        if rule is None or not note_assessment_text or not icd:
            return
        v = self.v
        anchor_words, anchor_low = v._note_evidence(note_assessment_text)

        mandated: set[str] = set()
        if v.store is not None:
            for e in icd:
                c = (e.get("code") or "").strip().upper()
                if not c:
                    continue
                for groups in (v.store.use_additional_code_groups(c),
                               v.store.code_also_groups(c)):
                    for _carrier, refs in groups:
                        mandated.update(r for r, _line in refs)
                mandated.update(v.store.code_first_etiology_refs(c))
        has_injury = any((e.get("code") or "").strip().upper()[:1] in ("S", "T")
                         for e in icd)
        markers = rule["residual_markers"]
        ev = rule.get("anchor_evidence", {})
        d_min = int(ev.get("desc_min_token_len", 3))
        i_min = int(ev.get("index_min_token_len", 4))

        def _documented_in_anchor(code: str, desc: str) -> bool:
            toks = [t for t in v._tokens(desc)
                    if len(t) >= d_min and t not in v._DESC_STOPWORDS]
            if toks:
                n = sum(1 for t in toks
                        if v._desc_documented(t, anchor_words, anchor_low))
                if n * 2 > len(toks):
                    return True
            if v.store is not None:
                norm = code.replace(".", "")
                for term, _c, s in v._index_rows():
                    if not norm.startswith(s):
                        continue
                    t_toks = [t for t in v._tokens(term)
                              if len(t) >= i_min
                              and t not in v._DESC_STOPWORDS]
                    if t_toks and all(
                            v._desc_documented(t, anchor_words, anchor_low)
                            for t in t_toks):
                        return True
            return False

        act = rule["action"]
        demoted = []
        for entry in icd:
            code = (entry.get("code") or "").strip().upper()
            norm = code.replace(".", "")
            if not norm:
                continue
            if str(entry.get("type", "")).strip().lower() == "primary":
                continue
            if str(entry.get("source_section", "")).startswith("validator:"):
                continue
            if norm[0] in ("V", "W", "X", "Y") and has_injury:
                continue  # external-cause companion of a billed injury
            if any(norm.startswith(m) or m.startswith(norm) for m in mandated):
                continue
            desc = ((v.db.validate_icd10(code) or {}).get("description") or "")
            dl = desc.lower()
            if not any(m in dl for m in markers):
                continue  # specific-entity codes are the sibling checks' turf
            if _documented_in_anchor(code, desc):
                continue
            demoted.append(entry)
        if not demoted:
            return
        icd[:] = [e for e in icd if e not in demoted]
        supporting = coding_result.setdefault("supporting_conditions", [])
        for entry in demoted:
            code = entry.get("code", "")
            entry["billable_tier"] = "advisory"
            entry["needs_review"] = True
            entry["review_reason"] = act["review_reason"]
            supporting.append(entry)
            v._add(
                act["severity"], code, act["category"],
                act["message"].format(
                    code=code, desc=(entry.get("description") or "")[:60]),
                act["recommendation"], denial_risk=act["denial_risk"],
            )
