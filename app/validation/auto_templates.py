"""Self-authored rule-engine templates (LLM-designed executor code).

The declarative rule engine's templates were hand-written Python mechanics;
rules are config those mechanics interpret. When the actuation proposer
escalates a flip class because NO existing template's mechanic fits, the
template synthesizer (tools/auto_actuate.py) asks the reasoning model to
DESIGN the missing template — as a small Python module. That code reaches
this loader only after passing deterministic gates: the AST safety
validation below, the no-hardcoded-medical-codes scan, and the full replay
gates (convergence / no-harm / inertness / registry protection) applied to
the first rule that uses it.

Contract of a template module (one file in data/rules/auto_templates/):

    TEMPLATE_NAME : str   snake_case name rules reference in "template"
    SCHEMA_DOC    : str   rule-schema documentation appended to the
                          proposer's system prompt so future flip classes
                          can be resolved with rules targeting it
    def execute(engine, rule, icd, cpt, hcpcs, coding_result,
                note_full_text, note_assessment_text) -> None

Execution environment is deliberately tiny: `re` plus a restricted
builtins set. Everything else a template needs arrives through `engine`
(the RuleEngine, whose .v is the CodingValidator: issue reporter,
language helpers, suppression set, reference DB, compliance store).
"""

from __future__ import annotations

import ast
import builtins
import re
from pathlib import Path

from loguru import logger

from app.core.config import DATA_DIR

AUTO_TEMPLATES_DIR = DATA_DIR / "rules" / "auto_templates"

_EXECUTE_PARAMS = ("engine", "rule", "icd", "cpt", "hcpcs", "coding_result",
                   "note_full_text", "note_assessment_text")

_ALLOWED_IMPORTS = {"re"}

# Names that must never be referenced: dynamic execution, I/O, reflection
# that could reach dunder machinery, interpreter control.
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
    "exit", "quit", "help", "dir", "type", "super", "object",
    "memoryview", "bytearray", "classmethod", "staticmethod", "property",
}

_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "int", "isinstance", "issubclass",
    "len", "list", "map", "max", "min", "next", "range", "repr",
    "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "ValueError", "KeyError", "IndexError", "TypeError",
    "AttributeError", "StopIteration", "ZeroDivisionError", "RuntimeError",
)
def _safe_import(name, *args, **kwargs):
    """Import hook honoring the same allowlist the static gate enforces —
    without it, the sandbox's builtins would make the (permitted)
    `import re` statement itself fail at load time."""
    if name in _ALLOWED_IMPORTS:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"import {name!r} is not allowed in auto templates")


SAFE_BUILTINS = {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES}
SAFE_BUILTINS["__import__"] = _safe_import

# Same lexical classes the rule gates forbid: CPT (5 digits), HCPCS
# (letter + 4 digits), dotted ICD-10-CM. A template must be fully generic
# — codes live in rule config selectors' GRAMMAR, never in executor code.
_CODE_LITERAL_RE = re.compile(
    r"\b\d{5}\b|\b[A-Z]\d{4}\b|\b[A-TV-Z]\d{2}\.\d{1,4}\b")

_MAX_AST_NODES = 4000

_FORBIDDEN_NODES = {
    ast.While: "while loops are forbidden (unbounded iteration) — iterate "
               "over finite collections with for",
    ast.With: "with blocks are forbidden (resource/file handles)",
    ast.AsyncWith: "async constructs are forbidden",
    ast.AsyncFor: "async constructs are forbidden",
    ast.AsyncFunctionDef: "async constructs are forbidden",
    ast.Await: "async constructs are forbidden",
    ast.Global: "global statements are forbidden",
    ast.Nonlocal: "nonlocal statements are forbidden",
    ast.ClassDef: "class definitions are forbidden — plain functions only",
}


def validate_template_source(src: str) -> list[str]:
    """Deterministic static gate for LLM-authored template code. Returns
    the list of violations (empty = safe to load). This is a whitelist
    posture: anything outside the tiny approved surface is a defect."""
    problems: list[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        problems.append(f"module too large ({len(nodes)} AST nodes, "
                        f"cap {_MAX_AST_NODES})")

    template_name = schema_doc = None
    execute_def = None

    def _norm_scan(text: str):
        """The code-literal regex over escape-normalized text."""
        return _CODE_LITERAL_RE.search(
            re.sub(r"\\[A-Za-z]", " ", re.sub(r"\\+\.", ".", text)))

    def _const_str(node):
        """Statically-known string value of an expression, or None.
        Catches literals the raw-source scan can't see as one token:
        implicit adjacency ('117' '20' folds to one Constant), explicit
        concatenation ('117' + '20'), and f-strings of constant parts
        (dynamic holes break adjacency with a space)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(v.value if isinstance(v, ast.Constant)
                           and isinstance(v.value, str) else " "
                           for v in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = _const_str(node.left), _const_str(node.right)
            if a is not None and b is not None:
                return a + b
        return None

    for node in nodes:
        folded = _const_str(node)
        if folded and len(folded) < 4096:
            m = _norm_scan(folded)
            if m:
                problems.append(
                    f"line {getattr(node, 'lineno', '?')}: literal medical "
                    f"code {m.group()!r} in a string constant expression")
        for cls, why in _FORBIDDEN_NODES.items():
            if isinstance(node, cls):
                problems.append(f"line {node.lineno}: {why}")
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in _ALLOWED_IMPORTS:
                    problems.append(f"line {node.lineno}: import "
                                    f"{a.name!r} not allowed (only "
                                    f"{sorted(_ALLOWED_IMPORTS)})")
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

    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            if node.targets[0].id == "TEMPLATE_NAME" and \
                    isinstance(node.value, ast.Constant) and \
                    isinstance(node.value.value, str):
                template_name = node.value.value
            elif node.targets[0].id == "SCHEMA_DOC" and \
                    isinstance(node.value, ast.Constant) and \
                    isinstance(node.value.value, str):
                schema_doc = node.value.value
        elif isinstance(node, ast.FunctionDef) and node.name == "execute":
            execute_def = node

    if template_name is None:
        problems.append("missing top-level TEMPLATE_NAME string constant")
    elif not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", template_name):
        problems.append(f"bad TEMPLATE_NAME {template_name!r} "
                        f"(snake_case, 3-41 chars)")
    if schema_doc is None:
        problems.append("missing top-level SCHEMA_DOC string constant")
    if execute_def is None:
        problems.append("missing top-level def execute(...)")
    else:
        args = tuple(a.arg for a in execute_def.args.args)
        if args != _EXECUTE_PARAMS:
            problems.append(
                f"execute() signature must be exactly "
                f"{_EXECUTE_PARAMS}, got {args}")
        if (execute_def.args.vararg or execute_def.args.kwarg
                or execute_def.args.kwonlyargs):
            problems.append("execute() may not take *args/**kwargs/"
                            "keyword-only args")

    # Recursion is the remaining unbounded-execution vector once while
    # loops are banned — reject any call cycle among the module's own
    # functions (direct or mutual).
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = {name: {c.func.id for c in ast.walk(fn)
                    if isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Name) and c.func.id in fns}
             for name, fn in fns.items()}

    def _reaches(start: str, target: str, seen: set) -> bool:
        for callee in calls.get(start, ()):
            if callee == target or (callee not in seen
                                    and _reaches(callee, target,
                                                 seen | {callee})):
                return True
        return False

    for name in fns:
        if _reaches(name, name, {name}):
            problems.append(f"function {name!r} participates in a "
                            f"recursion cycle — unbounded execution")
            break

    # Hardcoded medical codes: scan the source with the same escaped-dot
    # normalization the rule gate uses, so a regex like M77\.41 is caught
    # as the dotted ICD literal it is.
    m = _CODE_LITERAL_RE.search(
        re.sub(r"\\[A-Za-z]", " ", re.sub(r"\\+\.", ".", src)))
    if m:
        problems.append(f"literal medical code {m.group()!r} in template "
                        f"source — templates must be fully generic")
    return problems


def template_name_of(src: str) -> str:
    """The module's TEMPLATE_NAME constant, or '' — for callers that need
    the name before installing the file (the synthesizer's gates)."""
    try:
        for node in ast.parse(src).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "TEMPLATE_NAME"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                return node.value.value
    except SyntaxError:
        pass
    return ""


# path -> (mtime, entry). Entries are re-validated + re-executed whenever
# the file changes; a removed file simply stops being globbed.
_cache: dict[str, tuple[float, dict]] = {}


def _exec_module(src: str, path: Path) -> dict | None:
    ns = {"re": re, "__builtins__": SAFE_BUILTINS}
    exec(compile(src, str(path), "exec"), ns)  # noqa: S102 — gated above
    name, doc, fn = ns.get("TEMPLATE_NAME"), ns.get("SCHEMA_DOC"), \
        ns.get("execute")
    if not (isinstance(name, str) and isinstance(doc, str) and callable(fn)):
        return None
    return {"name": name, "schema_doc": doc, "execute": fn,
            "path": str(path)}


def load_auto_templates() -> dict[str, dict]:
    """{template_name: {name, schema_doc, execute, path}} for every
    template module that passes the static gate and loads cleanly.
    Defective files are skipped with a warning, never fatal — a broken
    auto template must degrade to 'rules referencing it are skipped'
    (and the pack audit then disables those rules)."""
    out: dict[str, dict] = {}
    if not AUTO_TEMPLATES_DIR.exists():
        return out
    for f in sorted(AUTO_TEMPLATES_DIR.glob("*.py")):
        try:
            mtime = f.stat().st_mtime
            cached = _cache.get(str(f))
            if cached and cached[0] == mtime:
                out[cached[1]["name"]] = cached[1]
                continue
            src = f.read_text(encoding="utf-8")
            problems = validate_template_source(src)
            if problems:
                logger.warning(f"Auto template {f.name} rejected by the "
                               f"safety gate: {problems[:3]} — skipped")
                continue
            entry = _exec_module(src, f)
            if entry is None:
                logger.warning(f"Auto template {f.name}: missing exports "
                               f"after execution — skipped")
                continue
            _cache[str(f)] = (mtime, entry)
            out[entry["name"]] = entry
        except Exception as exc:
            logger.warning(f"Auto template {f.name} failed to load: "
                           f"{exc!r} — skipped")
    return out
