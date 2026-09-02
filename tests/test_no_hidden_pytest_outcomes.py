"""Repository contract forbidding hidden pytest non-execution outcomes."""

from __future__ import annotations

import ast
from pathlib import Path


_TEST_ROOT = Path(__file__).resolve().parent
_PROHIBITED_PYTEST_CALLS = frozenset({"skip", "xfail", "importorskip"})
_PROHIBITED_MARKS = frozenset({"skip", "skipif", "xfail"})


def _attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    """Return a dotted-name chain for inert attribute syntax."""
    parts: list[str] = []
    cursor = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if not isinstance(cursor, ast.Name):
        return None
    parts.append(cursor.id)
    return tuple(reversed(parts))


def _hidden_outcomes(path: Path) -> list[str]:
    """Return source locations that can make a collected test silently non-executing."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if chain is None:
            continue
        direct = len(chain) == 2 and chain[0] == "pytest" and chain[1] in _PROHIBITED_PYTEST_CALLS
        marked = (
            len(chain) == 3
            and chain[:2] == ("pytest", "mark")
            and chain[2] in _PROHIBITED_MARKS
        )
        if direct or marked:
            findings.append(f"{path.relative_to(_TEST_ROOT)}:{node.lineno}:{'.'.join(chain)}")
    return findings


def test_owned_tests_do_not_hide_non_execution_with_pytest_skip_or_xfail() -> None:
    """Owned tests must execute, fail, or assert explicit capability absence rather than skip."""
    findings: list[str] = []
    for path in sorted(_TEST_ROOT.rglob("test_*.py")):
        if path == Path(__file__).resolve():
            continue
        findings.extend(_hidden_outcomes(path))
    assert findings == [], "hidden pytest outcomes:\n" + "\n".join(findings)
