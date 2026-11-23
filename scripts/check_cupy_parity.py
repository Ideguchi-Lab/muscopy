"""Post a PR reminder when a changed JAX function has no corresponding CuPy edit."""

# ruff: noqa: INP001

from __future__ import annotations

import ast
import json
import os
import subprocess  # noqa: S404
from pathlib import Path
from typing import TypedDict

# A single JAX function may correspond to a CuPy wrapper and one or more helpers.
# Shared code such as MuParameters already applies to both backends.
PARITY_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "dh": {
        "make_disk": ("make_disk", "_make_disk"),
        "crop_array": ("crop_array", "_crop_array"),
        "get_spectrum": ("get_spectrum", "_get_spectrum"),
        "get_spectrums": ("get_spectrums", "_get_spectrum"),
        "correct_aberration": ("correct_aberration", "_correct_aberration"),
        "offaxis_dh": ("offaxis_dh", "_offaxis_dh", "_as_single_center", "_validate_center"),
    },
    "phase_noise": {
        "calc_visibility": ("calc_visibility",),
        "_get_dc_ac": ("calc_visibility",),
        "_get_visibility": ("calc_visibility",),
    },
    "qpi": {
        "correct_phase_offset": ("correct_phase_offset",),
        "mip_qpi": ("mip_qpi",),
        "qpi": ("qpi",),
    },
    "qpi_utils": {
        "unwrap_phase": ("unwrap_phase",),
        "_wraptopi": ("_wraptopi",),
        "_solve_poisson": ("_solve_poisson",),
        "_dct2": ("_dct2",),
        "_idct2": ("_idct2",),
    },
}
COMMENT_MARKER = "<!-- muscopy-cupy-parity -->\n"


class _PullRequestHead(TypedDict):
    sha: str
    repo: dict[str, int] | None


class _PullRequest(TypedDict):
    number: int
    state: str
    base: dict[str, str]
    head: _PullRequestHead


class _Comment(TypedDict):
    id: int
    body: str
    user: dict[str, str]


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), encoding="utf-8")  # noqa: S603, S607


def _github_api(repository: str, endpoint: str, *args: str) -> str:
    return subprocess.check_output(  # noqa: S603
        ("gh", "api", f"repos/{repository}/{endpoint}", *args),  # noqa: S607
        encoding="utf-8",
    )


def _functions(revision: str, path: str) -> dict[str, list[str]]:
    if not _git("ls-tree", revision, "--", path):
        return {}

    definitions: dict[str, list[str]] = {}
    for node in ast.parse(_git("show", f"{revision}:{path}"), filename=path).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None:
                node.body = node.body[1:]
            definitions.setdefault(node.name, []).append(ast.dump(node))
    return definitions


def _changed_functions(base: str, head: str, path: str) -> set[str]:
    before = _functions(base, path)
    after = _functions(head, path)
    return {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}


def _review_items(base: str, head: str) -> list[str]:
    changed_paths = set(_git("diff", "--name-only", "--no-renames", "-z", base, head, "--", "muscopy").split("\0"))
    items = []
    for module, rules in PARITY_RULES.items():
        jax_path = f"muscopy/{module}.py"
        if jax_path not in changed_paths:
            continue
        cuda_path = f"muscopy/cuda/{module}.py"
        jax_changes = _changed_functions(base, head, jax_path)
        cuda_changes = _changed_functions(base, head, cuda_path) if cuda_path in changed_paths else set()
        for jax_name, cuda_names in rules.items():
            if jax_name in jax_changes and cuda_changes.isdisjoint(cuda_names):
                counterparts = ", ".join(f"`{name}()`" for name in cuda_names)
                items.append(f"- `{jax_path}` `{jax_name}()` → `{cuda_path}` {counterparts}")
    return items


def _update_comment(repository: str, number: int, items: list[str]) -> None:
    endpoint = f"issues/{number}/comments"
    pages: list[list[_Comment]] = json.loads(_github_api(repository, endpoint, "--paginate", "--slurp"))
    existing = next(
        (
            comment
            for page in pages
            for comment in page
            if comment["user"]["login"] == "github-actions[bot]" and comment["body"].startswith(COMMENT_MARKER)
        ),
        None,
    )
    if items:
        summary = "\n".join(items)
        body = (
            f"{COMMENT_MARKER}"
            "### CuPy parity review\n\n"
            "These JAX functions changed without a matching CuPy function edit in this PR.\n\n"
            f"{summary}\n\n"
            "Please review the CuPy implementation and update it if the JAX change applies."
        )
    elif existing is not None:
        body = f"{COMMENT_MARKER}### CuPy parity review\n\nThe reminder is cleared for the current PR diff."
    else:
        return

    if existing is None:
        _github_api(repository, endpoint, "--method", "POST", "--raw-field", f"body={body}", "--silent")
    elif existing["body"] != body:
        comment_endpoint = f"issues/comments/{existing['id']}"
        _github_api(repository, comment_endpoint, "--method", "PATCH", "--raw-field", f"body={body}", "--silent")


def _main() -> None:
    repository = os.environ["GITHUB_REPOSITORY"]
    run = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))["workflow_run"]
    head_repo = run["head_repository"]
    if head_repo is None or run["head_branch"] is None:
        return

    # The workflow_run payload can omit pull_requests for PRs from forks.
    pages: list[list[_PullRequest]] = json.loads(
        _github_api(
            repository,
            "pulls",
            "--method",
            "GET",
            "--raw-field",
            "state=open",
            "--raw-field",
            f"head={head_repo['owner']['login']}:{run['head_branch']}",
            "--paginate",
            "--slurp",
        )
    )
    for page in pages:
        for pr in page:
            head = pr["head"]["sha"]
            pr_head_repo = pr["head"]["repo"]
            if pr_head_repo is None or pr_head_repo["id"] != head_repo["id"] or head != run["head_sha"]:
                continue

            number = pr["number"]
            base = pr["base"]["sha"]
            # Read PR files from Git objects without checking out the PR or running its code.
            head_ref = f"refs/cupy-parity/{number}"
            refspec = f"+refs/pull/{number}/head:{head_ref}"
            _git("fetch", "--no-tags", "--no-recurse-submodules", "origin", base, refspec)
            if _git("rev-parse", head_ref).strip() != head:
                continue
            merge_base = _git("merge-base", base, head).strip()
            items = _review_items(merge_base, head)
            latest: _PullRequest = json.loads(_github_api(repository, f"pulls/{number}"))
            if latest["state"] == "open" and latest["head"]["sha"] == head and latest["base"]["sha"] == base:
                _update_comment(repository, number, items)


if __name__ == "__main__":
    _main()
