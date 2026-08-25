"""Test doubles for the SHORTLIST ENTAILMENT CONTRACT both judging calls answer.

`claude_coder.verify.select_entailed` and `.corroborate` each return a verdict about
EVERY shortlisted candidate: which ones the documentation still entails, and the NAMED
reason each of the others is out. That is what lets the resolver establish that the code
it releases is the ONLY one the documentation supports — two models agreeing about one
candidate never eliminated the rest (Codex F8-R1).

A stub answering with a bare `{"choice": n}` therefore models a judge that did NOT answer
the contract, and the resolver correctly refuses to treat such a selection as unique. That
is deliberate and is exercised directly in `tests/test_tie_policy.py`; every OTHER fixture
that means "this judge finds exactly these options entailed" should say so through the
helpers here rather than hand-assembling the JSON, so a future change to the contract has
ONE place to update instead of twenty.
"""
import json
import re

_OPTION = re.compile(r"(?m)^(\d+)\.\s+(.*)$")
#: issue #6 F9-R6: the "option N, <requirement_id>: axis=..." lines `verify.
#: _requirement_options` renders. Matches up to the literal `": axis="` marker, NOT
#: the next colon -- `requirement_id` itself is `f"{axis}:{code}:{n}"` and contains
#: colons, so a naive `[^:]+` would truncate it to just the axis name.
_REQUIREMENT_LINE = re.compile(r"(?m)^option \d+, (.+?): axis=")


def options(user: str) -> list:
    """The numbered candidate descriptors a judging call was actually shown."""
    block = user.split("CANDIDATE OFFICIAL DESCRIPTORS:", 1)[-1]
    return [(int(n), d) for n, d in _OPTION.findall(block)]


def requirement_ids(user: str) -> list:
    """The requirement_ids a judging call was actually shown, in order (issue #6
    F9-R6) — empty for any shortlist with no compiled requirements, matching
    `verify._shortlist_prompt`'s byte-identical-when-empty behavior."""
    parts = user.split("REQUIREMENTS:", 1)
    if len(parts) < 2:
        return []
    body = parts[1].split("\n\n", 1)[0]
    return [m.group(1) for m in _REQUIREMENT_LINE.finditer(body)]


def evidence_ids(user: str) -> list:
    """The bracketed evidence ids (`[e1]`, `[e2]`, ...) a judging call was actually
    shown, in order — only present once a shortlist carries compiled requirements
    (see `verify._evidence_options`); empty otherwise."""
    block = user.split("CANDIDATE OFFICIAL DESCRIPTORS", 1)[0]
    return re.findall(r"\[(e\d+)\]", block)


def verdict(user: str, entails=None, prefer=None, pick=None, reason="stub",
            missing_element=False, declare=True, requirement_status=None,
            requirement_quote="stub quote") -> str:
    """One judging model's answer over the shortlist it was shown.

    `entails(descriptor) -> bool` decides which options this judge finds entailed; `pick`
    (a 1-based option number) is the shorthand for "only this one". `prefer(descriptor)`
    chooses which entailed option it would code. Everything else is eliminated WITH a named
    reason. `declare=False` returns the bare pick a pre-contract model would have given —
    an answer that cannot establish uniqueness.

    `requirement_status` (issue #6 F9-R6): `{prefix: status}` — `prefix` is matched
    against the ACTUAL `requirement_id`s the shortlist showed (real ids look like
    `"<axis>:<candidate_code>:<n>"`; a prefix like `"laterality:CAND_RIGHT"` matches
    whichever real id that compiled to, so a test never has to predict the exact
    compiled id). A supported/contradicted status cites the first evidence id the
    shortlist actually showed (real behavior, not a fabricated span). Omit (default)
    to answer the shortlist contract only, unchanged from before this field existed.
    """
    opts = options(user)
    if entails is not None:
        entailed = [n for n, d in opts if entails(d)]
    elif pick is not None:
        entailed = [n for n, _ in opts if n == pick]
    else:
        entailed = [n for n, _ in opts][:1]
    if pick is not None:
        chosen = pick if pick in entailed else 0
    elif prefer is not None:
        chosen = next((n for n, d in opts if n in entailed and prefer(d)),
                      entailed[0] if entailed else 0)
    else:
        chosen = entailed[0] if entailed else 0
    if not declare:
        return json.dumps({"choice": chosen, "reason": reason})
    payload = {
        "choice": chosen,
        "entailed": entailed,
        "eliminated": [{"option": n,
                        "reason": "stub: not entailed by this documentation",
                        "missing_element": bool(missing_element)}
                       for n, _ in opts if n not in entailed],
        "reason": reason}
    if requirement_status:
        eids = evidence_ids(user)
        cite = [eids[0]] if eids else []
        shown = requirement_ids(user)
        matched = []
        for rid in shown:
            status = next((s for prefix, s in requirement_status.items()
                          if rid == prefix or rid.startswith(prefix + ":")), None)
            if status is None:
                continue
            matched.append({"requirement_id": rid, "status": status,
                            "span_ids": (cite if status in ("supported", "contradicted")
                                        else []),
                            "quote": (requirement_quote
                                     if status in ("supported", "contradicted") else "")})
        payload["requirements"] = matched
    return json.dumps(payload)


def judge(entails=None, prefer=None, pick=None, propose=(), reason="stub",
          missing_element=False, declare=True, requirement_status=None,
          requirement_quote="stub quote"):
    """A complete LLM callable: answers the PROPOSE prompt with `propose`, and BOTH judging
    prompts (selection and corroboration — they share one contract) with `verdict(...)`."""
    def stub(system, user):
        if "propose" in system.lower():
            return json.dumps({"codes": list(propose)})
        return verdict(user, entails=entails, prefer=prefer, pick=pick, reason=reason,
                       missing_element=missing_element, declare=declare,
                       requirement_status=requirement_status,
                       requirement_quote=requirement_quote)
    return stub


#: Nothing on the shortlist is entailed — the conservative answer.
NOTHING_ENTAILED = '{"choice": 0, "entailed": [], "reason": "stub: nothing entailed"}'
