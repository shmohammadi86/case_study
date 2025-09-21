import nox

PYTHON = "3.12"

@nox.session(python=PYTHON)
def format(session: nox.Session) -> None:
    session.run("uv", "sync", "--frozen", external=True)
    session.run("uv", "run", "ruff", "format", ".", external=True)

@nox.session(python=PYTHON)
def lint(session: nox.Session) -> None:
    session.run("uv", "sync", "--frozen", external=True)
    session.run("uv", "run", "ruff", "check", ".", external=True)

@nox.session(python=PYTHON)
def typecheck(session: nox.Session) -> None:
    session.run("uv", "sync", "--frozen", external=True)
    session.run("uv", "run", "mypy", "src", external=True)

@nox.session(python=PYTHON)
def tests(session: nox.Session) -> None:
    session.run("uv", "sync", "--frozen", external=True)
    session.run("uv", "run", "pytest", "-q", external=True)
