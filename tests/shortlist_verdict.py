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


def options(user: str) -> list:
    """The numbered candidate descriptors a judging call was actually shown."""
    block = user.split("CANDIDATE OFFICIAL DESCRIPTORS:", 1)[-1]
    return [(int(n), d) for n, d in _OPTION.findall(block)]


def verdict(user: str, entails=None, prefer=None, pick=None, reason="stub",
            missing_element=False, declare=True) -> str:
    """One judging model's answer over the shortlist it was shown.

    `entails(descriptor) -> bool` decides which options this judge finds entailed; `pick`
    (a 1-based option number) is the shorthand for "only this one". `prefer(descriptor)`
    chooses which entailed option it would code. Everything else is eliminated WITH a named
    reason. `declare=False` returns the bare pick a pre-contract model would have given —
    an answer that cannot establish uniqueness.
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
    return json.dumps({
        "choice": chosen,
        "entailed": entailed,
        "eliminated": [{"option": n,
                        "reason": "stub: not entailed by this documentation",
                        "missing_element": bool(missing_element)}
                       for n, _ in opts if n not in entailed],
        "reason": reason})


def judge(entails=None, prefer=None, pick=None, propose=(), reason="stub",
          missing_element=False, declare=True):
    """A complete LLM callable: answers the PROPOSE prompt with `propose`, and BOTH judging
    prompts (selection and corroboration — they share one contract) with `verdict(...)`."""
    def stub(system, user):
        if "propose" in system.lower():
            return json.dumps({"codes": list(propose)})
        return verdict(user, entails=entails, prefer=prefer, pick=pick, reason=reason,
                       missing_element=missing_element, declare=declare)
    return stub


#: Nothing on the shortlist is entailed — the conservative answer.
NOTHING_ENTAILED = '{"choice": 0, "entailed": [], "reason": "stub: nothing entailed"}'
