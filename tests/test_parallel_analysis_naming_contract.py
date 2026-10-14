"""Pin that ``parallel.rs`` means Horn's parallel analysis, not threads (#2009).

``crates/mlsirm-core/src/parallel.rs`` keeps its name to avoid import churn,
so its rustdoc and ``ARCHITECTURE.md`` must disambiguate it and send readers
of the CPU-parallelism track (#2001/#2002/#2003) to the bifactor E-step person
sweep instead.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARALLEL_RS = ROOT / "crates" / "mlsirm-core" / "src" / "parallel.rs"
BIFACTOR_GRM_RS = ROOT / "crates" / "mlsirm-core" / "src" / "bifactor_grm.rs"
ARCHITECTURE = ROOT / "ARCHITECTURE.md"

# Thread/data-parallel primitives whose presence would falsify the disclaimer.
_PARALLEL_PRIMITIVES = re.compile(
    r"\brayon\b|\bpar_iter\b|\bpar_chunks\b|\binto_par\b|std::thread|"
    r"\bthread::|available_parallelism"
)


def _module_doc(text: str) -> str:
    """Return the leading ``//!`` module documentation as one string."""
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("//!"):
            lines.append(line[3:].strip())
        elif lines or line.strip():
            break
    return " ".join(lines)


def _code_lines(text: str) -> list[str]:
    """Return non-comment source lines (``//`` comments stripped)."""
    return [line.split("//", 1)[0] for line in text.splitlines()]


def test_parallel_rs_module_doc_names_horn_and_disclaims_thread_parallelism() -> None:
    """The module banner must say Horn's PA and that it is not thread parallelism."""
    doc = _module_doc(PARALLEL_RS.read_text(encoding="utf-8"))
    assert doc.startswith("Horn's parallel analysis")
    assert "not thread" in doc.lower() or "no thread" in doc.lower()
    for issue in ("#2001", "#2002", "#2003"):
        assert issue in doc, issue
    assert "`crate::bifactor_grm::e_step`" in doc


def test_parallel_rs_code_uses_no_thread_or_data_parallel_primitives() -> None:
    """The disclaimer stays true: the module body spawns no threads."""
    code = "\n".join(_code_lines(PARALLEL_RS.read_text(encoding="utf-8")))
    assert _PARALLEL_PRIMITIVES.search(code) is None
    assert re.search(r"^pub fn parallel_analysis\(", code, re.MULTILINE)


def test_named_e_step_person_sweep_exists() -> None:
    """The pointer target must be a real function containing the person loop."""
    source = BIFACTOR_GRM_RS.read_text(encoding="utf-8")
    start = source.index("pub(crate) fn e_step(")
    end = source.index("\nfn ", start)
    assert "for p in 0..v.n_persons {" in source[start:end]


def test_architecture_glossary_disambiguates_parallel_rs() -> None:
    """The canonical architecture document carries the same disambiguation."""
    text = ARCHITECTURE.read_text(encoding="utf-8")
    paragraphs = [block for block in text.split("\n\n") if "src/parallel.rs" in block]
    assert len(paragraphs) == 1, "ARCHITECTURE.md must have one parallel.rs entry"
    flat = " ".join(paragraphs[0].split())
    assert "Horn's parallel analysis" in flat
    assert "not thread" in flat.lower()
    assert "`bifactor_grm::e_step`" in flat
    assert "#2002" in flat
