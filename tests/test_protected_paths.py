"""Tests for the protected-paths classifier and the files it depends on (design 8.1 to 8.3)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sol_eval.integrity import paths as pp

ROOT = Path(__file__).resolve().parents[1]
GLOBS = pp.load_protected_globs(ROOT / ".github" / "protected-paths")
OWNER = "minoya1220"
HEAD = "a" * 40
OLD = "b" * 40


def files(*names: str, renamed_from: dict[str, str] | None = None) -> list[dict]:
    renamed_from = renamed_from or {}
    return [
        {"filename": n, "status": "renamed" if n in renamed_from else "modified",
         "previous_filename": renamed_from.get(n)}
        for n in names
    ]


def review(state: str, commit: str = HEAD, login: str = OWNER, at: str = "2026-11-24T10:00:00Z", rid: int = 1):
    return {"id": rid, "state": state, "commit_id": commit, "submitted_at": at, "login": login}


def decide(fs, reviews=()):
    return pp.decide(list(fs), list(reviews), head_sha=HEAD, owner=OWNER, protected_globs=GLOBS)


# --- the files themselves -------------------------------------------------


def test_protected_paths_file_is_exactly_design_8_1():
    design = (ROOT / "docs" / "eval-design.md").read_text(encoding="utf-8")
    section = design.split("### 8.1 Protected paths", 1)[1].split("### 8.2", 1)[0]
    block = re.search(r"```\n(.*?)```", section, re.S).group(1)
    assert tuple(line for line in block.splitlines() if line.strip()) == GLOBS


def test_codeowners_assigns_every_protected_glob_to_the_owner():
    lines = [
        ln.split() for ln in (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert [ln[0] for ln in lines] == list(GLOBS)
    assert all(ln[1:] == [f"@{OWNER}"] for ln in lines)


def _workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_protected_paths_workflow_shape():
    wf = _workflow("protected-paths.yml")
    assert re.search(r"^on:\n  pull_request_target:\n    types: \[opened, synchronize, reopened, ready_for_review\]\n",
                     wf, re.M)
    assert "pull_request_review" not in wf.split("on:", 1)[1].split("permissions:", 1)[0]
    # checks out only the base commit, and nothing from the PR
    assert wf.count("uses: actions/checkout") == 1
    assert "ref: ${{ github.sha }}" in wf
    assert "persist-credentials: false" in wf
    assert "pulls/$PR_NUMBER/files" in wf and "--paginate" in wf and "previous_filename" in wf
    assert "pulls/$PR_NUMBER/reviews" in wf
    assert "name: protected-paths" in wf and "touches-protected" in wf


@pytest.mark.parametrize("name", ["protected-paths.yml", "ci.yml"])
def test_workflows_never_interpolate_pr_controlled_text(name):
    wf = _workflow(name)
    for field in ("head.ref", "head.label", "title", "body", "head.repo", "head_ref"):
        assert not re.search(r"\$\{\{[^}]*" + re.escape(field), wf), field
    assert "pull_request.head.sha" not in wf.replace("EVENT_HEAD_SHA: ${{ github.event.pull_request.head.sha }}", "")


def test_ci_workflow_is_cpu_read_only():
    wf = _workflow("ci.yml")
    assert "pull_request_target" not in wf
    assert re.search(r"^permissions:\n  contents: read\n\n", wf, re.M)
    assert not re.search(r"\$\{\{[^}]*secrets\.", wf)
    assert "name: lint" in wf and "name: unit" in wf
    assert "uv sync --frozen" in wf and "uv run pytest" in wf


# --- glob matching --------------------------------------------------------


@pytest.mark.parametrize(
    ("glob", "path", "hit"),
    [
        (".github/**", ".github/workflows/ci.yml", True),
        (".github/**", ".github/CODEOWNERS", True),
        (".github/**", "x/.github/a", False),
        ("**/conftest.py", "conftest.py", True),
        ("**/conftest.py", "a/b/conftest.py", True),
        ("**/conftest.py", "a/b/myconftest.py", False),
        ("**/*.pth", "src/evil.pth", True),
        ("src/sol_kernels/ops/**/*.py", "src/sol_kernels/ops/add.py", True),
        ("src/sol_kernels/ops/**/*.py", "src/sol_kernels/ops/a/b/add.py", True),
        ("src/sol_kernels/ops/**/*.py", "src/sol_kernels/ops/add.so", False),
        ("src/sol_kernels/ops/**/*.py", "src/sol_kernels/opsx/add.py", False),
        ("tuning/*.csv", "tuning/add.csv", True),
        ("tuning/*.csv", "tuning/sub/add.csv", False),
        ("pyproject.toml", "pyproject.toml", True),
        ("pyproject.toml", "sub/pyproject.toml", False),
    ],
)
def test_glob_match(glob, path, hit):
    assert pp.glob_match(glob, path) is hit


@pytest.mark.parametrize("bad", ["", "/abs", "dir/", "a/**b/c"])
def test_bad_globs_rejected(bad):
    with pytest.raises(ValueError):
        pp.glob_to_regex(bad)


# --- classification -------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "cls"),
    [
        ("src/sol_kernels/ops/add.py", "kernel"),
        ("src/sol_kernels/ops/elementwise/add.py", "kernel"),
        ("src/sol_kernels/templates/reduction.py", "kernel"),
        ("src/sol_kernels/cute_utils.py", "kernel"),
        ("tuning/softmax.csv", "kernel"),
        ("docs/ops/softmax.md", "kernel"),
        ("src/sol_kernels/ops/add.so", "other"),
        ("src/sol_kernels/ops/data.bin", "other"),
        ("tuning/softmax.json", "other"),
        ("src/sol_kernels/__init__.py", "other"),
        ("README.md", "other"),
        ("newdir/x.py", "other"),
        ("src/sol_kernels/ops/a b.py", "other"),
        ("src/sol_kernels/ops/a\\..\\x.py", "other"),
        (".github/workflows/ci.yml", "protected"),
        ("docs/eval-design.md", "protected"),
        ("pyproject.toml", "protected"),
        ("uv.lock", "protected"),
        (".python-version", "protected"),
        ("src/sol_kernels/_runtime/dispatch.py", "protected"),
        ("src/sol_eval/measure.py", "protected"),
        ("eval/exploits/E0001-torch-fallback/impl.py", "protected"),
        ("tests/test_x.py", "protected"),
        ("SRC/Sol_Eval/measure.py", "protected"),  # case-insensitive file systems
        ("tests/conftest.py", "protected"),
        ("src/sol_eval/__main__.py", "protected"),
        ("conftest.py", "forbidden"),
        ("src/sol_kernels/ops/conftest.py", "forbidden"),
        ("src/sol_kernels/ops/ConfTest.py", "forbidden"),
        ("src/sol_kernels/ops/__main__.py", "forbidden"),
        ("src/sol_kernels/__main__.py", "forbidden"),
        ("sitecustomize.py", "forbidden"),
        ("src/usercustomize.py", "forbidden"),
        ("src/sol_kernels/evil.pth", "forbidden"),
        ("tuning/x.PTH", "forbidden"),
    ],
)
def test_classify(path, cls):
    assert pp.classify(path, GLOBS) == cls


# --- decisions ------------------------------------------------------------


def test_kernel_only_passes_without_approval():
    d = decide(files("src/sol_kernels/ops/add.py", "tuning/add.csv"))
    assert d.passed and not d.needs_approval and not d.touches_protected


def test_empty_pr_passes():
    assert decide([]).passed


def test_protected_without_review_fails():
    d = decide(files("src/sol_eval/measure.py"))
    assert not d.passed and d.needs_approval and d.touches_protected


def test_protected_approved_on_head_passes():
    d = decide(files("src/sol_eval/measure.py"), [review("APPROVED")])
    assert d.passed and d.approved


def test_owner_login_compares_without_case():
    assert decide(files("src/sol_eval/measure.py"), [review("APPROVED", login="Minoya1220")]).passed


def test_approval_on_old_commit_fails():
    d = decide(files("src/sol_eval/measure.py"), [review("APPROVED", commit=OLD)])
    assert not d.passed and "not the head" in d.reasons[0]


def test_approval_by_someone_else_fails():
    assert not decide(files("src/sol_eval/measure.py"), [review("APPROVED", login="bot")]).passed


@pytest.mark.parametrize("later", ["COMMENTED", "CHANGES_REQUESTED", "DISMISSED"])
def test_newest_owner_review_must_be_the_approval(later):
    rs = [review("APPROVED", at="2026-11-24T10:00:00Z", rid=1), review(later, at="2026-11-24T11:00:00Z", rid=2)]
    assert not decide(files("src/sol_eval/measure.py"), rs).passed


def test_older_comment_does_not_hide_newer_approval():
    rs = [review("COMMENTED", at="2026-11-24T09:00:00Z", rid=1), review("APPROVED", at="2026-11-24T10:00:00Z", rid=2)]
    assert decide(files("src/sol_eval/measure.py"), rs).passed


def test_pending_review_is_ignored():
    rs = [review("APPROVED", rid=1), {"id": 2, "state": "PENDING", "commit_id": HEAD, "submitted_at": None, "login": OWNER}]
    assert decide(files("src/sol_eval/measure.py"), rs).passed


def test_other_path_needs_approval():
    assert not decide(files("README.md")).passed
    assert decide(files("README.md"), [review("APPROVED")]).passed


def test_kernel_plus_other_needs_approval():
    assert not decide(files("src/sol_kernels/ops/add.py", "README.md")).passed
    assert decide(files("src/sol_kernels/ops/add.py", "README.md"), [review("APPROVED")]).passed


def test_kernel_plus_protected_always_fails():
    d = decide(files("src/sol_kernels/ops/add.py", "src/sol_eval/errors.py"), [review("APPROVED")])
    assert not d.passed and any("split the harness change" in r for r in d.reasons)


def test_rename_counts_old_path():
    fs = files("src/sol_kernels/ops/errors.py", renamed_from={"src/sol_kernels/ops/errors.py": "src/sol_eval/errors.py"})
    d = decide(fs, [review("APPROVED")])
    assert not d.passed and d.touches_protected


def test_forbidden_file_fails_even_when_approved():
    d = decide(files("conftest.py"), [review("APPROVED")])
    assert not d.passed and d.reasons[0].startswith("FORBIDDEN_FILE")


def test_autorun_name_in_harness_dir_fails_in_a_kernel_pr():
    d = decide(files("src/sol_kernels/ops/add.py", "tests/conftest.py"), [review("APPROVED")])
    assert not d.passed and any(r.startswith("FORBIDDEN_FILE") for r in d.reasons)


def test_autorun_name_in_harness_dir_is_approvable_in_a_harness_pr():
    assert decide(files("tests/conftest.py"), [review("APPROVED")]).passed


# --- judge: checks on the fresh PR object ----------------------------------


def pr_obj(base="main", head=HEAD, changed=1):
    return {"base": {"ref": base}, "head": {"sha": head}, "changed_files": changed}


def judge(pr, fs, reviews=(), event_head=HEAD):
    return pp.judge(pr, list(fs), list(reviews), owner=OWNER, default_branch="main",
                    event_head_sha=event_head, protected_globs=GLOBS)


def test_judge_passes_kernel_pr():
    assert judge(pr_obj(), files("src/sol_kernels/ops/add.py")).passed


def test_judge_fails_on_non_default_base():
    d = judge(pr_obj(base="harness-protection"), files("src/sol_kernels/ops/add.py"))
    assert not d.passed and "retarget" in d.reasons[0]


def test_judge_fails_when_head_moved():
    assert not judge(pr_obj(), files("src/sol_kernels/ops/add.py"), event_head=OLD).passed


def test_judge_fails_when_file_list_is_truncated():
    assert not judge(pr_obj(changed=3001), files("src/sol_kernels/ops/add.py")).passed


def test_cli_judge_round_trip(tmp_path):
    (tmp_path / "pr.json").write_text(json.dumps(pr_obj(changed=2)))
    (tmp_path / "files.ndjson").write_text(
        "\n".join(json.dumps(f) for f in files("src/sol_eval/errors.py", "README.md")) + "\n")
    (tmp_path / "reviews.ndjson").write_text(json.dumps(review("APPROVED")) + "\n")
    out = tmp_path / "decision.json"
    # run the file the way the workflow does: plain python3, by path, from the repo root
    proc = subprocess.run(
        [sys.executable, "-I", "src/sol_eval/integrity/paths.py", "judge",
         "--pr", str(tmp_path / "pr.json"), "--files", str(tmp_path / "files.ndjson"),
         "--reviews", str(tmp_path / "reviews.ndjson"), "--owner", OWNER, "--default-branch", "main",
         "--event-head-sha", HEAD, "--protected-paths", ".github/protected-paths", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    d = json.loads(out.read_text())
    assert d["passed"] is True and d["touches_protected"] is True
    assert "protected-paths: PASS" in proc.stdout
