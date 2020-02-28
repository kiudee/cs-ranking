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

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version="1.0.2",
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            "numpy>=1.12.1",
            "scipy>=0.19.0",
            "scikit-learn>=0.18.2",
            "scikit-optimize>=0.4",
            "pandas>=0.22",
            "h5py>=2.7",
            "pygmo>=2.7",
            "docopt>=0.6.0",
            "joblib>=0.9.4",
            "tqdm>=4.11.2",
            "keras>=2.3",
            "pymc3>=3.8",
            "theano>=1.0",
            # Pick either CPU or GPU version of tensorflow:
            "tensorflow>=1.5,<2.0",
            # tensorflow-gpu>=1.0.1"
        ],
        extras_require={
            "data": [
                "psycopg2-binary>=2.7",  # database access
                "pandas>=0.22",
                "h5py>=2.7",
                "pygmo>=2.7",
            ],
        },
        package_data={"notebooks": ["*"]},
        include_package_data=True,
        long_description=readme + "\n\n" + history
    )
