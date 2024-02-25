from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install
import os

print(find_packages())

setup(
    name='graspness',
    version='1.0',
    packages=find_packages(),
    author='Your Name',
    author_email='your@email.com',
    description='Description of your library',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=["pointnet2_ops"]
)
