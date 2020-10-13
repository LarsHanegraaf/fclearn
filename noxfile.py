"""Nox sessions."""

import nox
import nox_poetry.patch  # noqa: F401

locations = ["fclearn", "tests", "noxfile.py"]
nox.options.sessions = "lint", "test"


@nox.session(python=["3.6"])
def test(session) -> None:
    """Test using pytest and pytest --cov."""
    session.install(".")
    session.install("pytest")
    session.install("pytest-cov")
    session.run("pytest", "--cov")


@nox.session(python=["3.6"])
def lint(session):
    """Lint using flake8."""
    args = session.posargs or locations
    session.install(
        "flake8", "flake8-black", "flake8-import-order", "flake8-docstrings", "darglint"
    )
    session.run("flake8", *args)


@nox.session(python=["3.6"])
def black(session):
    """Run black code formatter."""
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)
