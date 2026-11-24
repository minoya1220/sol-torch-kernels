"""Classify the paths a pull request changes and decide the `protected-paths` check.

Design: docs/eval-design.md sections 8.1 to 8.3 and threat H3 (2.1).

This module is run by `.github/workflows/protected-paths.yml` with the runner's own
`python3`, from a checkout of the base branch, never of the PR. So it must stay
standard-library only, must not import anything from `sol_eval` or `sol_kernels`,
and must run on Python 3.11 or later. The static lint (`integrity/lint.py`) reuses
it for `FORBIDDEN_FILE`.

Path classes:

- `kernel`: the only paths a kernel PR may touch (8.2).
- `protected`: matches a glob in `.github/protected-paths` (8.1). Matched without
  regard to case, so a path that would land on a protected file on a
  case-insensitive file system still counts as protected.
- `forbidden`: a file Python or pytest runs on its own (`conftest.py`,
  `sitecustomize.py`, `usercustomize.py`, `*.pth`, `__main__.py`) outside the
  harness directories (H3). No approval makes such a PR pass.
- `other`: everything else, including paths with unusual characters; needs the
  owner's approval (8.2).

Decision rules (8.2, 8.3):

- any `forbidden` path: fail (`FORBIDDEN_FILE`);
- an auto-run file name anywhere in a PR that also touches `kernel` paths: fail
  (`FORBIDDEN_FILE`), because H3 rejects these names anywhere in a kernel PR;
- `kernel` and `protected` paths in one PR: fail, approval or not;
- any `protected` or `other` path: pass only if the newest submitted review from
  the owner's account is `APPROVED` and was made on the PR's current head SHA;
- otherwise pass.

A rename counts both its old and its new path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

KERNEL_GLOBS: tuple[str, ...] = (
    "src/sol_kernels/ops/**/*.py",
    "src/sol_kernels/templates/**/*.py",
    "src/sol_kernels/cute_utils.py",
    "tuning/*.csv",
    "docs/ops/**",
)

# File names that Python's `site` module, pytest or `python -m` execute without an
# explicit import (H3). Matched on the last path segment, without regard to case.
AUTORUN_GLOBS: tuple[str, ...] = (
    "conftest.py",
    "sitecustomize.py",
    "usercustomize.py",
    "*.pth",
    "__main__.py",
)

SAFE_PATH = re.compile(r"^[A-Za-z0-9_.\-/]+$")

LABEL = "touches-protected"


# ---------------------------------------------------------------------------
# glob matching


def _segment_regex(segment: str) -> str:
    out = []
    for ch in segment:
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def glob_to_regex(pattern: str) -> str:
    """Translate a path glob to a regex. `*` and `?` stay inside one segment; a
    segment that is exactly `**` matches zero or more whole segments."""
    if not pattern or pattern.startswith("/") or pattern.endswith("/"):
        raise ValueError(f"bad glob {pattern!r}")
    parts = pattern.split("/")
    out = ""
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if part == "**":
            out += ".*" if last else "(?:[^/]+/)*"
        else:
            if "**" in part:
                raise ValueError(f"'**' must be a whole segment in {pattern!r}")
            out += _segment_regex(part) + ("" if last else "/")
    return "^" + out + "$"


def glob_match(pattern: str, path: str, *, ignore_case: bool = False) -> bool:
    flags = re.IGNORECASE if ignore_case else 0
    return re.match(glob_to_regex(pattern), path, flags) is not None


def load_protected_globs(path: str | Path) -> tuple[str, ...]:
    """One glob per line. Blank lines and lines starting with '#' are skipped."""
    globs = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        glob_to_regex(line)  # validate
        globs.append(line)
    if not globs:
        raise ValueError(f"{path}: no globs")
    return tuple(globs)


# ---------------------------------------------------------------------------
# classification


def is_autorun_name(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return any(glob_match(g, name, ignore_case=True) for g in AUTORUN_GLOBS)


def _harness_globs(protected_globs: tuple[str, ...]) -> tuple[str, ...]:
    """Protected globs that name harness directories or files, as opposed to the
    `**/conftest.py`-style globs that exist only to catch auto-run names."""
    return tuple(
        g for g in protected_globs
        if not (g.startswith("**/") and is_autorun_name(g[3:].replace("*", "x")))
    )


def matches_protected(path: str, protected_globs: tuple[str, ...]) -> bool:
    return any(glob_match(g, path, ignore_case=True) for g in protected_globs)


def classify(path: str, protected_globs: tuple[str, ...]) -> str:
    """Return 'forbidden', 'protected', 'kernel' or 'other' for one repo path."""
    if is_autorun_name(path):
        in_harness = any(glob_match(g, path, ignore_case=True) for g in _harness_globs(protected_globs))
        return "protected" if in_harness else "forbidden"
    if matches_protected(path, protected_globs):
        return "protected"
    if not SAFE_PATH.match(path) or any(seg in ("", ".", "..") for seg in path.split("/")):
        return "other"
    if any(glob_match(g, path) for g in KERNEL_GLOBS):
        return "kernel"
    return "other"


def forbidden_files(paths: list[str], protected_globs: tuple[str, ...]) -> list[tuple[str, str]]:
    """The `FORBIDDEN_FILE` part of the decision, for a PR's changed paths: an
    auto-run file outside the harness, or an auto-run file anywhere in a PR that
    touches kernel paths (H3). Returns (path, rule) pairs. Shared with the lint."""
    classes = {p: classify(p, protected_globs) for p in paths}
    kernel_pr = "kernel" in classes.values()
    out: list[tuple[str, str]] = []
    for p in paths:
        if classes[p] == "forbidden":
            out.append((p, "autorun_outside_harness"))
        elif kernel_pr and is_autorun_name(p):
            out.append((p, "autorun_in_kernel_pr"))
    return out


# ---------------------------------------------------------------------------
# decision


@dataclass
class Decision:
    passed: bool
    touches_protected: bool
    needs_approval: bool
    approved: bool
    reasons: list[str] = field(default_factory=list)
    paths: dict[str, list[str]] = field(default_factory=dict)  # class -> paths


def newest_owner_review(reviews: list[dict], owner: str) -> dict | None:
    """Newest submitted review by `owner` (GitHub logins compare without case).
    Pending reviews have no `submitted_at` and are ignored."""
    mine = [
        r for r in reviews
        if (r.get("login") or "").lower() == owner.lower() and r.get("submitted_at")
    ]
    if not mine:
        return None
    # ISO 8601 UTC strings from the API sort correctly as text; id breaks ties.
    return max(mine, key=lambda r: (r["submitted_at"], int(r.get("id") or 0)))


def decide(
    files: list[dict],
    reviews: list[dict],
    *,
    head_sha: str,
    owner: str,
    protected_globs: tuple[str, ...],
) -> Decision:
    by_class: dict[str, list[str]] = {"forbidden": [], "protected": [], "kernel": [], "other": []}
    touches_protected = False
    all_paths: list[str] = []
    for f in files:
        for p in (f.get("filename"), f.get("previous_filename")):
            if not p or p in all_paths:
                continue
            all_paths.append(p)
            by_class[classify(p, protected_globs)].append(p)
            if matches_protected(p, protected_globs):
                touches_protected = True

    reasons: list[str] = []
    hard_fail = False
    bad = forbidden_files(all_paths, protected_globs)
    outside = [p for p, rule in bad if rule == "autorun_outside_harness"]
    in_kernel_pr = [p for p in all_paths if by_class["kernel"] and is_autorun_name(p)]
    if outside:
        hard_fail = True
        reasons.append("FORBIDDEN_FILE: auto-executed file outside the harness: " + ", ".join(outside))
    if in_kernel_pr:
        hard_fail = True
        reasons.append(
            "FORBIDDEN_FILE: auto-executed file in a PR that touches kernel paths: " + ", ".join(in_kernel_pr)
        )
    if by_class["kernel"] and by_class["protected"]:
        hard_fail = True
        reasons.append(
            "kernel and protected paths in one PR: split the harness change into its own PR "
            f"(protected: {', '.join(by_class['protected'])})"
        )

    needs_approval = bool(by_class["protected"] or by_class["other"] or by_class["forbidden"])
    approved = False
    if needs_approval:
        review = newest_owner_review(reviews, owner)
        if review is None:
            reasons.append(f"needs an approving review from {owner} on {head_sha}; none found")
        elif review.get("state") != "APPROVED":
            reasons.append(f"newest review from {owner} is {review.get('state')}, not APPROVED")
        elif review.get("commit_id") != head_sha:
            reasons.append(
                f"newest approval from {owner} is on {review.get('commit_id')}, not the head {head_sha}"
            )
        else:
            approved = True

    passed = not hard_fail and (approved or not needs_approval)
    return Decision(
        passed=passed,
        touches_protected=touches_protected,
        needs_approval=needs_approval,
        approved=approved,
        reasons=reasons,
        paths={k: v for k, v in by_class.items() if v},
    )


def judge(
    pr: dict,
    files: list[dict],
    reviews: list[dict],
    *,
    owner: str,
    default_branch: str,
    event_head_sha: str,
    protected_globs: tuple[str, ...],
) -> Decision:
    """`decide` plus checks on the PR object read fresh from the API."""
    head_sha = pr["head"]["sha"]
    d = decide(files, reviews, head_sha=head_sha, owner=owner, protected_globs=protected_globs)
    pre: list[str] = []
    if pr["base"]["ref"] != default_branch:
        pre.append(
            f"PR targets {pr['base']['ref']!r}, not {default_branch!r}: this check is only "
            "meaningful from the default branch's copy of the workflow; retarget to "
            f"{default_branch!r}"
        )
    if head_sha != event_head_sha:
        pre.append(f"head moved from {event_head_sha} to {head_sha} during the run; this run is stale")
    expected = pr.get("changed_files")
    if not isinstance(expected, int) or expected != len(files):
        pre.append(
            f"file list incomplete: the API listed {len(files)} files, the PR reports {expected!r}"
        )
    if pre:
        d.passed = False
        d.reasons = pre + d.reasons
    return d


# ---------------------------------------------------------------------------
# command line (used by the workflow)


def _read_ndjson(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _summary(d: Decision) -> str:
    lines = [f"protected-paths: {'PASS' if d.passed else 'FAIL'}", ""]
    for cls, paths in d.paths.items():
        lines.append(f"{cls} ({len(paths)}):")
        lines.extend(f"  {p}" for p in paths)
    if d.reasons:
        lines.append("")
        lines.extend(f"- {r}" for r in d.reasons)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    j = sub.add_parser("judge", help="decide the protected-paths check for one PR")
    j.add_argument("--pr", required=True, help="JSON of GET /repos/{o}/{r}/pulls/{n}")
    j.add_argument("--files", required=True, help="NDJSON: filename, status, previous_filename")
    j.add_argument("--reviews", required=True, help="NDJSON: id, state, commit_id, submitted_at, login")
    j.add_argument("--owner", required=True)
    j.add_argument("--default-branch", required=True)
    j.add_argument("--event-head-sha", required=True)
    j.add_argument("--protected-paths", default=".github/protected-paths")
    j.add_argument("--out", required=True)

    c = sub.add_parser("classify", help="print the class of each path given")
    c.add_argument("paths", nargs="+")
    c.add_argument("--protected-paths", default=".github/protected-paths")

    args = ap.parse_args(argv)
    globs = load_protected_globs(args.protected_paths)
    if args.cmd == "classify":
        for p in args.paths:
            print(f"{classify(p, globs)}\t{p}")
        return 0

    d = judge(
        json.loads(Path(args.pr).read_text(encoding="utf-8")),
        _read_ndjson(args.files),
        _read_ndjson(args.reviews),
        owner=args.owner,
        default_branch=args.default_branch,
        event_head_sha=args.event_head_sha,
        protected_globs=globs,
    )
    Path(args.out).write_text(json.dumps(asdict(d), indent=2), encoding="utf-8")
    print(_summary(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
