from os.path import realpath, dirname, join

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

DISTNAME = "csrank"
DESCRIPTION = "Context-sensitive ranking"
MAINTAINER = "Karlson Pfannschmidt"
MAINTAINER_EMAIL = "kiudee@mail.upb.de"

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version="1.0.0",
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=install_reqs,
        package_data={"notebooks": ["*"]},
        include_package_data=True,
        long_description=readme + "\n\n" + history
    )
