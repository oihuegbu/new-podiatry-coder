"""Graduated self-authored templates — trusted, static, part of the app.

A template Fable designs at runtime starts life sandboxed in
data/rules/auto_templates/ (re-validated on every load). Once it has
PROVEN itself in production — deterministic criteria in
tools/graduate_templates.py: live long enough, executed across enough
fresh documents, no rule referencing it ever disabled, no flip class it
closed ever reopened — the graduation tool promotes its module verbatim
into this package. From then on it is application code: imported
statically, reviewed in code review like any app change, no runtime
sandbox — the same trust level as the hand-written mechanics in
rule_engine.py, which is exactly what graduation means.

Each module keeps the auto-template contract (TEMPLATE_NAME, SCHEMA_DOC,
execute(engine, rule, icd, cpt, hcpcs, coding_result, note_full_text,
note_assessment_text)) so rules referencing it keep working unchanged —
graduation moves the code's trust category, never its behavior.
"""

import importlib
import pkgutil

from loguru import logger

GRADUATED: dict[str, dict] = {}


def refresh() -> dict[str, dict]:
    """(Re)scan the package for graduated template modules. Called at
    import and by the graduation tool right after a promotion, so the
    template vocabulary never shrinks mid-process when the sandbox copy
    is retired."""
    GRADUATED.clear()
    importlib.invalidate_caches()  # a promotion may have just added a file
    for m in pkgutil.iter_modules(__path__):
        try:
            mod = importlib.import_module(f"{__name__}.{m.name}")
            mod = importlib.reload(mod)
            name = getattr(mod, "TEMPLATE_NAME", None)
            fn = getattr(mod, "execute", None)
            if isinstance(name, str) and callable(fn):
                GRADUATED[name] = {
                    "name": name,
                    "schema_doc": str(getattr(mod, "SCHEMA_DOC", "")),
                    "execute": fn,
                    "path": getattr(mod, "__file__", ""),
                }
            else:
                logger.warning(f"Graduated template module {m.name}: "
                               f"missing TEMPLATE_NAME/execute — ignored")
        except Exception as exc:  # a broken graduate must not kill the app
            logger.error(f"Graduated template module {m.name} failed to "
                         f"import: {exc!r} — ignored")
    return GRADUATED


refresh()
