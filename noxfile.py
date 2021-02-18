import  nox
from nox_poetry import session


locations = "csrank", "noxfile.py"


@session(python="3.7")
def tests(session):
    session.run("poetry", "install", external=True)
    session.run("pytest", "csrank/tests")
