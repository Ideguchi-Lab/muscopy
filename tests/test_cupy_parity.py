from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from scripts import check_cupy_parity as parity

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        (
            {
                "muscopy/dh.py": "def crop_array(): return 2\n",
                "muscopy/cuda/dh.py": "def _crop_array(): return 1\ndef make_disk(): return 2\n",
            },
            [
                (
                    "- `muscopy/dh.py` `crop_array()` → `muscopy/cuda/dh.py` "
                    "`crop_array()`, `_crop_array()`, `_crop_slices()`"
                )
            ],
        ),
        (
            {
                "muscopy/dh.py": "def crop_array(): return 2\n",
                "muscopy/cuda/dh.py": "def _crop_array(): return 2\ndef make_disk(): return 1\n",
            },
            [],
        ),
        (
            {"muscopy/phase_noise.py": "def _get_visibility(): return 2\ndef calc_phase_noise(): return 1\n"},
            ["- `muscopy/phase_noise.py` `_get_visibility()` → `muscopy/cuda/phase_noise.py` `calc_visibility()`"],
        ),
        (
            {
                "muscopy/phase_noise.py": "def _get_visibility(): return 1\ndef calc_phase_noise(): return 2\n",
                "muscopy/odt.py": "def odt(): return 1\n",
                "muscopy/idt.py": "def compute_idt(): return 1\n",
            },
            [],
        ),
        ({"muscopy/dh.py": 'def crop_array():\n    """Updated docstring."""\n    return 1\n'}, []),
    ],
    ids=["unrelated-cupy-edit", "matching-cupy-helper", "visibility-helper", "jax-only", "docstring-only"],
)
def test_review_items_for_pr_diff(
    updates: dict[str, str], expected: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    parity._git("init", "--quiet")  # noqa: SLF001
    parity._git("config", "user.name", "CuPy parity test")  # noqa: SLF001
    parity._git("config", "user.email", "test@example.invalid")  # noqa: SLF001

    files = {
        "muscopy/dh.py": "def crop_array(): return 1\n",
        "muscopy/cuda/dh.py": "def _crop_array(): return 1\ndef make_disk(): return 1\n",
        "muscopy/phase_noise.py": "def _get_visibility(): return 1\ndef calc_phase_noise(): return 1\n",
        "muscopy/cuda/phase_noise.py": "def calc_visibility(): return 1\n",
    }
    for filename, source in files.items():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    parity._git("add", ".")  # noqa: SLF001
    parity._git("commit", "--quiet", "-m", "Base")  # noqa: SLF001

    for filename, source in updates.items():
        (tmp_path / filename).write_text(source, encoding="utf-8")
    parity._git("add", ".")  # noqa: SLF001
    parity._git("commit", "--quiet", "-m", "PR change")  # noqa: SLF001

    assert parity._review_items("HEAD^", "HEAD") == expected  # noqa: SLF001


def test_existing_reminder_is_cleared_without_posting_another_comment(monkeypatch: pytest.MonkeyPatch) -> None:
    body = f"{parity.COMMENT_MARKER}Open reminder"
    api = Mock(return_value=json.dumps([[{"id": 30, "body": body, "user": {"login": "github-actions[bot]"}}]]))
    monkeypatch.setattr(parity, "_github_api", api)

    parity._update_comment("owner/repo", 12, [])  # noqa: SLF001

    assert api.call_count == 2
    assert api.call_args.args[1:4] == ("issues/comments/30", "--method", "PATCH")
    assert "reminder is cleared" in api.call_args.args[5]
