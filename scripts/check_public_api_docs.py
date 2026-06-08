#!/usr/bin/env python
"""Audit ``sleap_roots_analyze.__all__`` for introspection-readiness.

bloom-mcp's autopop generator builds MCP tool descriptors by reading
``sleap_roots_analyze.__all__`` and, for each name, calling ``inspect.signature()``,
``typing.get_type_hints()``, and ``__doc__``. This script enforces the contract that
makes that introspection succeed for every public symbol:

For each public **function**:
    1. Every parameter (excluding ``*args``/``**kwargs``) is annotated.
    2. The return value is annotated.
    3. ``typing.get_type_hints()`` resolves without raising.
    4. A docstring exists with a ``Returns:`` section, plus an ``Args:`` section
       when the function takes parameters.
    5. Every parameter name appears in the docstring body.
    6. A ``Raises:`` section is present when the function body raises a non-trivial
       exception (a ``raise SomeError(...)`` in its own body, not a bare re-raise).

For each public **class**:
    7. The class has a docstring.
    8. Every ``__init__`` parameter beyond ``self`` is annotated and named in the
       class or ``__init__`` docstring.

Run standalone (exits non-zero on any violation)::

    uv run python scripts/check_public_api_docs.py

Or import :func:`run_audit` (returns the list of violation strings) from a test.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
import typing

import sleap_roots_analyze as sra

_IGNORED_PARAMS = {"self", "cls"}
_VAR_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _real_params(sig: inspect.Signature) -> list[inspect.Parameter]:
    """Return the documentable parameters of a signature.

    Args:
        sig: Signature to inspect.

    Returns:
        Parameters excluding ``self``/``cls`` and ``*args``/``**kwargs``.
    """
    return [
        p
        for p in sig.parameters.values()
        if p.name not in _IGNORED_PARAMS and p.kind not in _VAR_KINDS
    ]


def _raises_nontrivially(func: object) -> bool:
    """Report whether a function's own body raises a non-trivial exception.

    A bare ``raise`` (re-raise) and ``raise`` statements inside nested functions or
    lambdas are ignored, so only exceptions the function itself originates count.

    Args:
        func: Function object to inspect.

    Returns:
        True if the function body contains a ``raise <Exception>(...)``.
    """
    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return False

    # The parsed module's first statement is the function definition itself.
    func_node = tree.body[0]
    found = False

    def visit(node: ast.AST, *, top: bool) -> None:
        nonlocal found
        # Do not descend into nested callables (their raises aren't this func's).
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if not top:
                return
        if isinstance(node, ast.Raise) and node.exc is not None:
            found = True
        for child in ast.iter_child_nodes(node):
            visit(child, top=False)

    visit(func_node, top=True)
    return found


def _check_function(name: str, func: object) -> list[str]:
    """Check one public function against the introspection contract.

    Args:
        name: The ``__all__`` name being checked.
        func: The function object.

    Returns:
        A list of violation messages (empty if the function passes).
    """
    problems: list[str] = []

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError) as exc:
        return [f"{name}: inspect.signature() failed ({exc})"]

    params = _real_params(sig)

    for p in params:
        if p.annotation is inspect.Parameter.empty:
            problems.append(f"{name}: parameter '{p.name}' has no type annotation")
    if sig.return_annotation is inspect.Signature.empty:
        problems.append(f"{name}: missing return annotation")

    try:
        typing.get_type_hints(func)
    except Exception as exc:
        # Intentionally broad: any failure to resolve hints (NameError, etc.)
        # is a violation we want to report rather than propagate.
        problems.append(
            f"{name}: typing.get_type_hints() failed " f"({type(exc).__name__}: {exc})"
        )

    doc = inspect.getdoc(func) or ""
    if not doc.strip():
        problems.append(f"{name}: missing docstring")
        return problems

    if params and "Args:" not in doc:
        problems.append(f"{name}: docstring missing 'Args:' section")
    if "Returns:" not in doc:
        problems.append(f"{name}: docstring missing 'Returns:' section")
    for p in params:
        if not re.search(rf"\b{re.escape(p.name)}\b", doc):
            problems.append(f"{name}: parameter '{p.name}' not documented")
    if _raises_nontrivially(func) and "Raises:" not in doc:
        problems.append(f"{name}: raises but docstring has no 'Raises:' section")

    return problems


def _check_class(name: str, cls: type) -> list[str]:
    """Check one public class against the introspection contract.

    Args:
        name: The ``__all__`` name being checked.
        cls: The class object.

    Returns:
        A list of violation messages (empty if the class passes).
    """
    problems: list[str] = []

    class_doc = inspect.getdoc(cls) or ""
    if not class_doc.strip():
        problems.append(f"{name}: class missing docstring")

    init = cls.__init__
    # An inherited ``object.__init__`` means no declared constructor params.
    if init is object.__init__:
        return problems

    try:
        sig = inspect.signature(init)
    except (ValueError, TypeError):
        return problems

    params = _real_params(sig)
    init_doc = inspect.getdoc(init) or ""
    combined_doc = f"{class_doc}\n{init_doc}"
    for p in params:
        if p.annotation is inspect.Parameter.empty:
            problems.append(
                f"{name}.__init__: parameter '{p.name}' has no type annotation"
            )
        if not re.search(rf"\b{re.escape(p.name)}\b", combined_doc):
            problems.append(f"{name}.__init__: parameter '{p.name}' not documented")

    return problems


def run_audit() -> list[str]:
    """Audit every ``sleap_roots_analyze.__all__`` entry.

    Returns:
        A flat list of violation messages across all public symbols. An empty list
        means the entire public API satisfies the introspection contract.
    """
    violations: list[str] = []
    for name in sra.__all__:
        obj = getattr(sra, name, None)
        if obj is None:
            violations.append(f"{name}: not found on sleap_roots_analyze")
            continue
        if inspect.isclass(obj):
            violations.extend(_check_class(name, obj))
        elif callable(obj):
            violations.extend(_check_function(name, obj))
        else:
            violations.append(f"{name}: not callable and not a class ({type(obj)})")
    return violations


def main() -> int:
    """Run the audit and print a per-symbol report.

    Returns:
        Process exit code: 0 if every public symbol passes, 1 otherwise.
    """
    violations = run_audit()
    total = len(sra.__all__)
    failing_names = {v.split(":", 1)[0].split(".", 1)[0] for v in violations}

    print(f"Public API introspection audit: {total} __all__ entries")
    print(f"  passing: {total - len(failing_names)}")
    print(f"  failing: {len(failing_names)}")
    if violations:
        print("\nViolations:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("\nAll public API entries are introspection-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
