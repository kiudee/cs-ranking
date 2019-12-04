from os.path import realpath, dirname, join

from setuptools import setup, find_packages
import versioneer

DISTNAME = 'csrank' # TODO
DESCRIPTION = 'Context-sensitive ranking'
MAINTAINER = 'Karlson Pfannschmidt'
MAINTAINER_EMAIL = 'kiudee@mail.upb.de'
VERSION = "1.0"

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

if __name__ == "__main__":
    setup(name=DISTNAME,
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          packages=find_packages(),
          install_requires=install_reqs,
          package_data={'notebooks': ['*']},
          include_package_data=True)
