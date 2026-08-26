# Commercial release stage deadlines

## Fixed

- Commercial release subprocesses remain fail closed and output-bounded, while the outer `release_acceptance.py` watchdog now has a 3,600-second stage budget that exceeds the current 3,360-second sum of its sequential inner acceptance deadlines. Cold `python -m build` remains bounded at 900 seconds and other release stages retain the 300-second default, preventing the outer orchestrator from terminating legitimate bounded acceptance work before the inner operation-specific deadlines can report their own failure evidence.
