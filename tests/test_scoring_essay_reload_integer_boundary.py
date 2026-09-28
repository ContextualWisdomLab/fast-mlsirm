"""Regression tests for essay integer safety across module reloads."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_contracts_reload_preserves_public_integer_boundary_in_isolation() -> None:
    """Implementation reload cannot weaken the public boundary or poison this test process."""
    script = textwrap.dedent(
        r'''
        import importlib

        import numpy as np

        from fast_mlsirm.scoring import AssessmentSpecError
        from fast_mlsirm.scoring import essay as essay_package
        from fast_mlsirm.scoring.essay import contracts as essay_contracts


        class HostileIndex:
            callbacks = 0

            def __index__(self):
                type(self).callbacks += 1
                return 5_000


        def prompt(maximum_response_characters):
            return essay_package.build_essay_prompt(
                prompt_id="argument_prompt",
                task_family_id="essay_review",
                prompt_content_fingerprint="1" * 64,
                language_id="english_language",
                genre_id="argument_genre",
                maximum_response_characters=maximum_response_characters,
                maximum_response_units=1_000,
            )


        importlib.reload(essay_contracts)

        try:
            prompt(HostileIndex())
        except AssessmentSpecError as exc:
            assert exc.code == "invalid_maximum_response_characters"
        else:
            raise AssertionError("hostile __index__ provider was accepted after reload")

        assert HostileIndex.callbacks == 0

        accepted = prompt(np.uint32(5_000))
        assert type(accepted.maximum_response_characters) is int
        assert accepted.maximum_response_characters == 5_000
        '''
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_public_factory_seals_integer_before_reentrant_contracts_reload() -> None:
    """A reload inside construction cannot regain caller-controlled integer coercion."""
    script = textwrap.dedent(
        r'''
        import importlib

        import numpy as np

        from fast_mlsirm.scoring import AssessmentSpecError
        from fast_mlsirm.scoring import essay as essay_package
        from fast_mlsirm.scoring.essay import contracts as essay_contracts


        class HostileIndex:
            callbacks = 0

            def __index__(self):
                type(self).callbacks += 1
                return 5_000


        reloads = 0
        original_descriptive_identifier = essay_contracts.descriptive_identifier


        def reload_on_first_identifier(value, name):
            global reloads
            if reloads == 0:
                reloads += 1
                importlib.reload(essay_contracts)
            return original_descriptive_identifier(value, name)


        essay_contracts.descriptive_identifier = reload_on_first_identifier


        def prompt(maximum_response_characters):
            return essay_package.build_essay_prompt(
                prompt_id="argument_prompt",
                task_family_id="essay_review",
                prompt_content_fingerprint="1" * 64,
                language_id="english_language",
                genre_id="argument_genre",
                maximum_response_characters=maximum_response_characters,
                maximum_response_units=1_000,
            )


        try:
            prompt(HostileIndex())
        except AssessmentSpecError as exc:
            assert exc.code == "invalid_maximum_response_characters"
        else:
            raise AssertionError("hostile __index__ provider crossed the public boundary")

        assert HostileIndex.callbacks == 0
        assert reloads == 0

        accepted = prompt(np.uint32(5_000))
        assert reloads == 1
        assert type(accepted.maximum_response_characters) is int
        assert accepted.maximum_response_characters == 5_000
        '''
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
