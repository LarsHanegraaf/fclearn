import nox
import nox_poetry.patch
from nox.sessions import Session
import os

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