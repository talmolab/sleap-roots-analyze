r"""Source guard: pipeline steps must not pre-stringify paths (issue #157).

Producers store ``Path`` objects in ``files_generated`` and path-valued
``metadata`` entries and let the central serializers normalize once
(``convert_to_json_serializable`` for ``pipeline_summary.json``; the
``save_json`` ``default`` hook for standalone manifests). A producer-side
``str(path)`` defeats that normalization and bakes in backslash separators on
Windows. This AST guard fails if the anti-pattern is reintroduced, so the fix
can't silently regress in a future step. It is multi-line aware (unlike a
line-oriented grep), so it catches ``files_generated=[\n    str(x),\n]`` too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

STEPS_DIR = Path(__file__).resolve().parents[1] / (
    "src/sleap_roots_analyze/pipeline/steps"
)

# Dict keys / subscript targets whose values are filesystem paths. A ``str()``
# wrapping the value of one of these is the #157 anti-pattern. Error-message and
# data-value keys (``error``, ``genotype``, ...) are intentionally excluded.
_PATH_KEY_HINTS = (
    "path",
    "csv",
    "json",
    "plot",
    "file",
    "files",
    "dir",
    "directory",
    "output",
    "barplot",
    "dashboard",
)


def _is_str_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
    )


def _looks_like_path_key(key: ast.AST) -> bool:
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        low = key.value.lower()
        return any(hint in low for hint in _PATH_KEY_HINTS)
    return False


def _step_files() -> list[Path]:
    return sorted(p for p in STEPS_DIR.glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("path", _step_files(), ids=lambda p: p.name)
def test_no_str_prestringify_of_paths(path: Path) -> None:
    """No ``str(...)`` feeds ``files_generated`` or a path-valued ``metadata`` key."""
    tree = ast.parse(path.read_text(), filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        # files_generated=[...] / files_generated=[..., str(x)]
        if isinstance(node, ast.keyword) and node.arg == "files_generated":
            for inner in ast.walk(node.value):
                if _is_str_call(inner):
                    violations.append(f"files_generated keyword at line {inner.lineno}")

        # files_generated.append(str(x))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "files_generated"
        ):
            for inner in ast.walk(node):
                if _is_str_call(inner):
                    violations.append(f"files_generated.append at line {node.lineno}")

        # {"...path...": str(x)} dict literal entries with a path-ish key
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    key is not None
                    and _looks_like_path_key(key)
                    and _is_str_call(value)
                ):
                    violations.append(
                        f"dict key {ast.literal_eval(key)!r} at line {value.lineno}"
                    )

        # metadata["...path..."] = str(x) subscript assignments
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and _looks_like_path_key(target.slice)
                    and _is_str_call(node.value)
                ):
                    violations.append(f"subscript assign at line {node.lineno}")

    assert not violations, (
        f"{path.name}: pre-stringified path(s) found — store Path, not str(path). "
        f"Sites: {violations}"
    )
