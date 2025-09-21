from pathlib import Path

import nox

PYTHON = "3.12"


@nox.session(python=PYTHON)
def format(session: nox.Session) -> None:
    # Install tooling into the session venv
    session.run("uv", "pip", "install", "ruff>=0.5.0", external=True)
    # Run formatter
    session.run("ruff", "format", ".", external=True)


@nox.session(python=PYTHON)
def lint(session: nox.Session) -> None:
    session.run("uv", "pip", "install", "ruff>=0.5.0", external=True)
    session.run("ruff", "check", ".", "--fix", external=True)


@nox.session(python=PYTHON)
def typecheck(session: nox.Session) -> None:
    session.run("uv", "pip", "install", "mypy>=1.8.0", external=True)
    # Collect files explicitly to avoid mypy directory resolution quirks
    files = [
        str(p)
        for p in Path(".").rglob("*.py")
        if ".nox" not in str(p) and ".venv" not in str(p) and "venv" not in str(p)
    ]
    if not files:
        session.error("No Python files found for type checking.")
        return
    session.run("mypy", *files, external=True)


@nox.session(python=PYTHON)
def tests(session: nox.Session) -> None:
    # Optional: basic pytest runner if tests exist
    session.run("uv", "pip", "install", "pytest>=8.0.0", external=True)
    session.run("pytest", "-q", external=True)
