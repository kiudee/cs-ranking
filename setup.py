from os.path import dirname
from os.path import realpath

from setuptools import find_packages
from setuptools import setup

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
        version="1.2.0",
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Natural Language :: English",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering",
            "Topic :: Software Development",
        ],
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            "numpy>=1.12.1",
            "scipy>=0.19.0",
            "scikit-learn>=0.18.2",
            "scikit-optimize>=0.4",
            "pandas>=0.22",
            "h5py>=2.7",
            "docopt>=0.6.0",
            "joblib>=0.9.4",
            "tqdm>=4.11.2",
            "keras>=2.3",
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
            "probabilistic": ["pymc3>=3.8", "theano>=1.0"],
        },
        package_data={"notebooks": ["*"]},
        include_package_data=True,
        long_description=readme + "\n\n" + history,
        url="https://github.com/kiudee/cs-ranking",
    )
