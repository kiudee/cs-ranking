"""Nox sessions."""
from pathlib import Path
import shutil
from textwrap import dedent

import nox
from nox_poetry import session

locations = "csrank", "noxfile.py"
nox.options.sessions = ("pre-commit", "tests", "docs-build")
python_versions = ["3.8"]


def activate_virtualenv_in_precommit_hooks(session):
    """Activate virtualenv in hooks installed by pre-commit.
    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.
    Args:
        session: The Session object.
    """
    if session.bin is None:
        return

    virtualenv = session.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        text = hook.read_text()
        bindir = repr(session.bin)[1:-1]  # strip quotes
        if not (
            Path("A") == Path("a") and bindir.lower() in text.lower() or bindir in text
        ):
            continue

        lines = text.splitlines()
        if not (lines[0].startswith("#!") and "python" in lines[0].lower()):
            continue

        header = dedent(
            f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """
        )

        lines.insert(1, header)
        hook.write_text("\n".join(lines))


@session(python=python_versions)
def tests(session):
    """Run the test suite."""
    session.install(".[data]")
    session.install("coverage[toml]", "pytest", "nox", "nox-poetry")
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage")


@session
def coverage(session):
    """Produce the coverage report."""
    # Do not use session.posargs unless this is the only session.
    nsessions = len(session._runner.manifest)
    has_args = session.posargs and nsessions == 1
    args = session.posargs if has_args else ["report"]

    session.install("coverage[toml]")

    if not has_args and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args, "-i")


@session(python="3.8")
def black(session):
    """Run black code formatter."""
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


@session(name="pre-commit", python="3.8")
def precommit(session):
    args = session.posargs or ["run", "--all-files", "--show-diff-on-failure"]
    session.install(
        "pre-commit",
        "black",
        "flake8",
        "doc8",
        "zimports",
    )
    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@session(name="docs-build", python="3.8")
def docs_build(session):
    """Build the documentation."""
    args = session.posargs or ["docs", "docs/_build"]
    session.install(".")
    session.install(
        "sphinx",
        "sphinx-autobuild",
        "sphinx_rtd_theme",
        "sphinxcontrib-bibtex",
        "nbsphinx",
        "IPython",
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)


@session(python="3.8")
def docs(session):
    """Build and serve the documentation."""
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session.install(".")
    session.install(
        "sphinx",
        "sphinx-autobuild",
        "sphinx_rtd_theme",
        "sphinxcontrib-bibtex",
        "nbsphinx",
        "IPython",
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
