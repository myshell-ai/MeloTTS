from setuptools import setup, find_packages
import os 

cwd = os.path.dirname(os.path.abspath(__file__))
requirements = open(os.path.join(cwd, "requirements.txt"), "r").readlines()

setup(
    name='MyShellTTSBase',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    package_data={
        '': ['*.txt', 'cmudict_*'],
    },
)
