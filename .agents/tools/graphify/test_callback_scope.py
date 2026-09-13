"""Regression for Python callback resolution across lexical scopes."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


FIXTURE_SOURCES = {
    "callbacks.py": """def imported_callback(source_value):
    return source_value
""",
    "main.py": """from callbacks import imported_callback

def dispatch(callback_function):
    return callback_function(None)

def handler(source_value):
    return source_value

def first_sibling():
    def handler(source_value):
        return source_value

    def closure(source_value):
        dispatch(handler)
        return source_value

    dispatch(handler)
    closure(None)

def second_sibling():
    def handler(source_value):
        return source_value

    def closure(source_value):
        dispatch(handler)
        return source_value

    dispatch(handler)
    closure(None)

def cross_file():
    dispatch(imported_callback)

def attribute_shadow(object_value):
    handler = None
    dispatch(getattr(object_value, "handler"))
""",
}


class CallbackScopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="graphify-1862-")
        cls.addClassCleanup(cls.temp_dir.cleanup)
        cls.fixture_dir = Path(cls.temp_dir.name) / "fixture"
        cls.output_dir = Path(cls.temp_dir.name) / "graph"
        cls.fixture_dir.mkdir()
        for filename, source_text in FIXTURE_SOURCES.items():
            (cls.fixture_dir / filename).write_text(source_text)
        command = [
            sys.executable,
            "-m",
            "graphify",
            "extract",
            str(cls.fixture_dir),
            "--out",
            str(cls.output_dir),
            "--no-cluster",
            "--max-workers",
            "1",
        ]
        process_result = subprocess.run(command, text=True, capture_output=True)
        cls.command_output = process_result.stdout + process_result.stderr
        if process_result.returncode:
            raise RuntimeError(
                f"Graphify extraction failed ({process_result.returncode}):\n{cls.command_output}"
            )
        graph_path = cls.output_dir / "graphify-out" / "graph.json"
        cls.edges = json.loads(graph_path.read_text())["edges"]

    def assert_edge(self, source_node, target_node, context="argument"):
        found = any(
            edge_record.get("source") == source_node
            and edge_record.get("target") == target_node
            and edge_record.get("relation") == "indirect_call"
            and edge_record.get("context") == context
            for edge_record in self.edges
        )
        self.assertTrue(
            found,
            f"missing {source_node} -> {target_node} ({context}); extraction output:\n{self.command_output}",
        )

    def assert_no_edge(self, source_node, target_node, context="argument"):
        found = any(
            edge_record.get("source") == source_node
            and edge_record.get("target") == target_node
            and edge_record.get("relation") == "indirect_call"
            and edge_record.get("context") == context
            for edge_record in self.edges
        )
        self.assertFalse(
            found,
            f"unexpected {source_node} -> {target_node} ({context}); extraction output:\n{self.command_output}",
        )

    def test_sibling_callbacks_bind_to_their_own_nested_handlers(self):
        self.assert_edge("main_first_sibling", "main_first_sibling_handler")
        self.assert_edge("main_first_sibling_closure", "main_first_sibling_handler")
        self.assert_edge("main_second_sibling", "main_second_sibling_handler")
        self.assert_edge("main_second_sibling_closure", "main_second_sibling_handler")
        self.assert_no_edge("main_first_sibling", "main_handler")
        self.assert_no_edge("main_second_sibling", "main_handler")

    def test_cross_file_callback_remains_resolvable(self):
        self.assert_edge("main_cross_file", "callbacks_imported_callback")

    def test_getattr_keeps_attribute_name_heuristic(self):
        self.assert_edge(
            "main_attribute_shadow", "main_handler", context="getattr"
        )
        self.assert_no_edge("main_attribute_shadow", "main_handler")


if __name__ == "__main__":
    unittest.main()
