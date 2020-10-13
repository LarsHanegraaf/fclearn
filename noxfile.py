import nox
import nox_poetry.patch  # noqa: F401

locations = ["fclearn", "tests", "noxfile.py"]
nox.options.sessions = "lint", "test"


@nox.session(python=["3.6"])
def test(session) -> None:
    """
    Run unit tests.

    Arguments:
        session: The Session object.
    """
    session.install(".")
    session.install("pytest")
    session.install("pytest-cov")
    session.run("pytest", "--cov")


@nox.session(python=["3.6"])
def lint(session):
    args = session.posargs or locations
    session.install("flake8", "flake8-black", "flake8-import-order")
    session.run("flake8", *args)


@nox.session(python=["3.6"])
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)
