"""The flag catalogue (design 2.6) and the result every check returns (design 9.5).

Each flag has a code, a severity and a one-line doc. Severities:

- REJECT: the impl is invalid for this op; no score is reported.
- FAIL: a case failed; the op does not pass, other cases are still scored.
- ENV: the measurement is invalid because of the machine; not the author's fault.
- WARN: reported, does not change the verdict.

Two flags have a different severity in CI than in evaluation (`TUNING_STALE`,
`TUNING_TABLE_TOO_LARGE`); `ci_severity` holds that. Rows of the 2.6 table that
name several codes (`MEMCHECK_ERROR, RACECHECK_ERROR, SYNCCHECK_ERROR`,
`WRONG_DTYPE, WRONG_SHAPE`, `CRASH, TIMEOUT`) are one member per code.

`evidence_keys` are the fixed evidence fields of a flag (9.5). Only the flags
raised by the checks that exist so far have them; the step that implements a
check fills in its flag's keys.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Severity(enum.Enum):
    REJECT = "REJECT"
    FAIL = "FAIL"
    ENV = "ENV"
    WARN = "WARN"


_LINT_KEYS = ("file", "line", "col", "rule", "detail")


class Flag(enum.Enum):
    # code, severity, doc, ci_severity, evidence_keys
    FORBIDDEN_IMPORT = ("REJECT", "Impl imports a module outside the allowlist.", None, _LINT_KEYS + ("module",))
    FORBIDDEN_CALL = ("REJECT", "Impl uses a banned builtin, attribute pattern, or audited call.", None, _LINT_KEYS)
    FORBIDDEN_FILE = ("REJECT", "PR adds an auto-executed file or touches a path kernel PRs may not touch.", None, ("file", "rule", "detail"))
    FOREIGN_KERNEL = ("REJECT", "A kernel, memcpy or memset not in the impl allowlist ran during a checked window.", None, ())
    HARNESS_TAMPERED = ("REJECT", "Tripwire saw a changed harness or torch function.", None, ())
    HIDDEN_WORK = ("REJECT", "Device work from the process overlapped the timed window outside the timed graph.", None, ())
    BUILD_LAUNCHED_KERNEL = ("REJECT", "Build launched device work.", None, ())
    ALLOC_IN_CALL = ("REJECT", "Build or launcher allocated device memory outside the workspace.", None, ())
    UNEXPECTED_THREAD = ("REJECT", "New OS threads after impl import, build or call.", None, ())
    STALE_OUTPUT = ("REJECT", "A checked output still holds poison or the previous replay's value.", None, ())
    TIMING_MISMATCH = ("REJECT", "Event timing, wall clock and trace kernel spans disagree beyond tolerance (4.8).", None, ())
    SOL_ABOVE_PEAK = ("REJECT", "Achieved bandwidth above 1.02x measured peak.", None, ())
    CANARY_CORRUPTED = ("REJECT", "Red zone bytes changed.", None, ())
    MEMCHECK_ERROR = ("REJECT", "compute-sanitizer memcheck reported an error.", None, ())
    RACECHECK_ERROR = ("REJECT", "compute-sanitizer racecheck reported an error.", None, ())
    SYNCCHECK_ERROR = ("REJECT", "compute-sanitizer synccheck reported an error.", None, ())
    PROTECTED_PATH_CHANGED = ("REJECT", "Protected tree hash differs from main (REJECT for a kernel PR).", None, ())
    WRONG_RESULT = ("FAIL", "Error outside the bound, or a class mismatch (NaN, inf, sign).", None, ())
    WRONG_DTYPE = ("FAIL", "Output dtype differs from torch's.", None, ())
    WRONG_SHAPE = ("FAIL", "Output shape differs from torch's.", None, ())
    MISSING_ERROR = ("FAIL", "Torch raises for this case and the impl neither refused nor raised.", None, ())
    NONDETERMINISTIC = ("FAIL", "Two runs of the same case differ bitwise.", None, ())
    CRASH = ("FAIL", "Exception on a supported spec.", None, ())
    TIMEOUT = ("FAIL", "Hang on a supported spec.", None, ())
    TUNING_STALE = ("WARN", "Tuning rows whose kernel hash does not match.", "FAIL", ())
    TUNING_TABLE_TOO_LARGE = ("FAIL", "Table over the row limit.", "FAIL", ())
    UNSUPPORTED = ("WARN", "supports() returned false; counted in coverage.", None, ())
    THROTTLED = ("ENV", "NVML clock-event reasons showed thermal, power or hardware slowdown in the window.", None, ())
    MEM_CLOCK_DRIFT = ("ENV", "Memory clock during the op differs from the clock during the peak measurement by more than 1%.", None, ())
    UNSTABLE_TIMING = ("ENV", "Spread of replays above the limit after retries (4.5).", None, ())
    FOREIGN_PROCESS = ("ENV", "Another compute process was on the GPU.", None, ())
    L2_RESIDENT = ("WARN", "Case too small to rotate out of L2; reported as latency only.", None, ())

    def __init__(self, severity: str, doc: str, ci_severity: str | None, evidence_keys: tuple[str, ...]) -> None:
        self.severity = Severity(severity)
        self.doc = doc
        self.ci_severity = Severity(ci_severity) if ci_severity else self.severity
        self.evidence_keys = evidence_keys

    @property
    def code(self) -> str:
        return self.name


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome (9.5). `case_id` is None for static checks, which are
    about files, not cases."""

    code: Flag
    passed: bool
    case_id: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.code, Flag):
            raise TypeError(f"code must be a Flag, got {self.code!r}")
        if self.code.evidence_keys:
            extra = set(self.evidence) - set(self.code.evidence_keys)
            if extra:
                raise ValueError(f"{self.code.code}: evidence keys {sorted(extra)} not in {self.code.evidence_keys}")

    def message(self) -> str:
        e = self.evidence
        where = e.get("file", "")
        if "line" in e:
            where += f":{e['line']}:{e.get('col', 0)}"
        return f"{where}: {self.code.code} [{e.get('rule', '')}] {e.get('detail', '')}".strip()
