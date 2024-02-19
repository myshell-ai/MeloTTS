import os 
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

cwd = os.path.dirname(os.path.abspath(__file__))
requirements = open(os.path.join(cwd, "requirements.txt"), "r").readlines()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        os.system('python -m unidic download')


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        os.system('python -m unidic download')

setup(
    name='melo',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    package_data={
        '': ['*.txt', 'cmudict_*'],
    },
)
