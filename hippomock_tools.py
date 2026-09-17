"""
Grounding tool for HippoMock test generation.

HippoMock is a small, header-only C++ mocking library with far less
representation in LLM training data than e.g. Google Mock — models routinely
confuse the two APIs. Rather than hard-coding a "cheat sheet" that might be
wrong, this tool locates the *actual* hippomocks.h the target project uses
and extracts a curated excerpt from it (macro definitions, key method
signatures, one worked example of each mocking style), so generated tests
are grounded in the real API instead of a guess.

A full vendored hippomocks.h (with template specializations for every call
arity) can run 300-400K+ characters — far too large to hand an LLM whole
without risking token-budget/rate-limit failures, and mostly repetitive
boilerplate anyway. So instead of returning the raw file, this searches it
for a fixed set of structurally-important anchors (verified against a real
copy of the header) and returns generous context around each match, with
the resolved file path included so the agent can read_file more of it
directly if something isn't covered.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.tools import BaseTool, tool

_FILENAME_CANDIDATES = (
    "hippomocks.h",  # actual upstream filename (github.com/dascandy/hippomock)
    "HippoMocks.h",
    "hippomocks.hpp",
    "hippomock.h",   # kept as fallback for older/renamed vendored copies
    "HippoMock.h",
    "hippomock.hpp",
)

# Directories (relative to project root) commonly used to vendor third-party
# headers — searched first since a vendored copy is the most authoritative
# version for this specific project.
_VENDOR_DIRNAMES = ("third_party", "thirdparty", "vendor", "external", "extern", "deps", "libs", "lib")

# Fallback global locations for common C++ package managers, only searched
# if nothing is found inside the project.
_GLOBAL_ROOTS = [
    Path(os.environ.get("VCPKG_ROOT", "")) if os.environ.get("VCPKG_ROOT") else None,
    Path(r"C:\vcpkg"),
    Path.home() / ".conan2" / "p",
    Path.home() / ".conan" / "data",
]

# (label, regex, lines of context after the match line, max matches to keep)
# Verified against a real vendored hippomocks.h. Patterns are generic
# (not tied to this file's specific line numbers) so they should hold across
# versions/forks that keep the same macro/method naming.
_ANCHOR_PATTERNS: list[tuple[str, str, int, int]] = [
    (
        "Mock/expectation macros (interface AND free-function mocking)",
        r"^#define\s+(On|Expect|Never)Call\w*\(",
        0,
        40,
    ),
    (
        "Free/static function mocking declaration (RegisterExpect_ for a bare function pointer)",
        r"RegisterExpect_\([A-Za-z0-9_ ]*\(\*func\)\(",
        0,
        3,
    ),
    (
        "Interface/virtual-method mocking declaration (RegisterExpect_ for obj + member-function pointer)",
        r"RegisterExpect_\([A-Za-z0-9_ ]*\*\w+,\s*[A-Za-z0-9_ ]+\(\w+::\*",
        0,
        3,
    ),
    (
        "Creating a mock object: MockRepository::Mock<T>()",
        r"base\s*\*\s*MockRepository::Mock\(\)",
        8,
        1,
    ),
    (
        "Verifying expectations were met: VerifyAll()",
        r"void\s+VerifyAll\(\)",
        12,
        1,
    ),
    (
        "Chaining an expectation: .With(...) / .Return(...) on the call object",
        r"^class TCall\b",
        45,
        1,
    ),
]

_HEADER_COMMENT_LINES = 60  # top-of-file license/config comments — small and useful
_MAX_CHARS = 40_000  # safety net; curated output should stay well under this


def _search(root: Path, max_depth: int) -> Path | None:
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - root_depth
        if depth >= max_depth:
            dirnames[:] = []  # don't descend further
            continue
        # Skip huge/irrelevant trees to keep the search fast.
        dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "__pycache__", ".codegraph")]
        for name in filenames:
            if name in _FILENAME_CANDIDATES:
                return Path(dirpath) / name
    return None


def _build_excerpt(text: str) -> str:
    lines = text.splitlines()
    sections: list[str] = [
        "--- Top-of-file comments (license, config options) ---\n"
        + "\n".join(lines[:_HEADER_COMMENT_LINES])
    ]

    # Collect (start, end, labels) ranges first, then merge overlapping/
    # adjacent ones before rendering — anchors with 0 lines of context
    # (e.g. the macro block, where every #define is its own match) would
    # otherwise print the same lines many times over as each match's
    # "one line of leading context" overlaps the previous match.
    ranges: list[tuple[int, int, list[str]]] = []
    not_found: list[str] = []
    for label, pattern, context_after, max_matches in _ANCHOR_PATTERNS:
        regex = re.compile(pattern, re.MULTILINE)
        found = 0
        for m in regex.finditer(text):
            if found >= max_matches:
                break
            line_no = text.count("\n", 0, m.start())
            start = max(0, line_no - 1)
            end = min(len(lines), line_no + context_after + 1)
            ranges.append((start, end, [label]))
            found += 1
        if found == 0:
            not_found.append(label)

    ranges.sort(key=lambda r: r[0])
    merged: list[tuple[int, int, list[str]]] = []
    for start, end, labels in ranges:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end, prev_labels = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_labels + labels)
        else:
            merged.append((start, end, labels))

    for start, end, labels in merged:
        snippet = "\n".join(lines[start:end])
        label_line = " / ".join(dict.fromkeys(labels))  # de-dupe, keep order
        sections.append(f"--- {label_line} (lines {start + 1}-{end}) ---\n{snippet}")

    for label in not_found:
        sections.append(f"--- {label}: NOT FOUND in this header ---")

    excerpt = "\n\n".join(sections)
    if len(excerpt) > _MAX_CHARS:
        excerpt = excerpt[:_MAX_CHARS] + "\n\n*** excerpt truncated at safety cap ***"
    return excerpt


def make_hippomock_tools(project_root: Path) -> list[BaseTool]:
    project_root = project_root.resolve()

    @tool
    def locate_hippomock(hint_path: str | None = None) -> str:
        """Find the project's actual hippomocks.h and return a grounded excerpt.

        Always call this before writing any HippoMock mock or test code, and
        base the generated code on the returned excerpt rather than on
        memorized syntax — HippoMock's API is easy to confuse with other
        mocking libraries (e.g. Google Mock), and getting it wrong produces
        code that won't compile. The excerpt covers BOTH mocking styles the
        library supports: interface/virtual-method mocking (MockRepository,
        Mock<T>(), ExpectCall/OnCall) and free/static function mocking
        (ExpectCallFunc/OnCallFunc), if present in this copy of the header.

        The excerpt is curated (not the full file, which can be 300K+ chars
        of repetitive template boilerplate) — it includes the resolved file
        path, so if you need to see more of the real header, use read_file
        on that path directly (only works if the header is inside the
        project root; sandboxed read_file can't reach a global install).

        Search order: an explicit hint_path if given, then vendored
        third-party directories inside the project, then the rest of the
        project tree, then common global package-manager install locations
        (vcpkg, conan) on this machine.

        Args:
            hint_path: Optional path (relative to the project root, or
                absolute) to hippomocks.h if you already know where it is.
        """
        candidates: list[Path] = []

        if hint_path:
            p = Path(hint_path)
            candidates.append(p if p.is_absolute() else project_root / p)

        for vendor_dir in _VENDOR_DIRNAMES:
            vd = project_root / vendor_dir
            if vd.is_dir():
                found = _search(vd, max_depth=6)
                if found:
                    candidates.append(found)

        found_in_project = _search(project_root, max_depth=8)
        if found_in_project:
            candidates.append(found_in_project)

        for global_root in _GLOBAL_ROOTS:
            if global_root and global_root.is_dir():
                found = _search(global_root, max_depth=5)
                if found:
                    candidates.append(found)

        for path in candidates:
            if path and path.is_file():
                content = path.read_text(encoding="utf-8", errors="replace")
                excerpt = _build_excerpt(content)
                return (
                    f"Found hippomock header at: {path} ({len(content)} chars total; "
                    f"showing a curated {len(excerpt)}-char excerpt below, not the "
                    "full file).\n\n" + excerpt
                )

        return (
            "hippomocks.h was not found in the project, its vendored "
            "third-party directories, or common global package-manager "
            "locations (vcpkg, conan) on this machine.\n\n"
            "Do not write HippoMock mock/EXPECT syntax from memory — its API "
            "is a frequent source of hallucination (it is often confused "
            "with Google Mock's MOCK_METHOD/EXPECT_CALL style). Instead:\n"
            "  1. Ask the user where hippomocks.h lives in their project, or\n"
            "  2. Ask them to point you at it via the hint_path argument, or\n"
            "  3. If it needs to be added to the project, tell the user to "
            "vendor it from https://github.com/dascandy/hippomock and call "
            "this tool again — do not fabricate its contents yourself."
        )

    return [locate_hippomock]
