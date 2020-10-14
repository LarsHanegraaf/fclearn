"""Nox sessions."""

import nox
import nox_poetry.patch  # noqa: F401

locations = ["fclearn", "tests", "noxfile.py", "docs/conf.py"]
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


@nox.session(python=["3.6"])
def docs(session):
    """Build the documentation."""
    session.install(".")
    session.install("sphinx", "sphinx-autodoc-typehints", "sphinx_rtd_theme")
    session.run(
        "sphinx-apidoc", "-o", "docs/source/", "fclearn", "-e", "-t", "docs/templates/"
    )
    session.run("sphinx-build", "docs", "docs/_build")
