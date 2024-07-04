# -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup

import nyx

VERSION = nyx.__version__
NAME = 'nyx'
DESCRIPTION = "Bath's Artificial Intelligence M.Sc. dissertation project."
AUTHOR = 'Gergely Tóth'
AUTHORS_EMAIL_ADDRESS = 'greg.toth@protonmail.com'
URL = 'https://github.bath.ac.uk/gt566/nyx.git'


def read(filename):
    """Read the content of a given filename"""
    _path = os.path.join(os.path.dirname(__file__), filename)
    return open(_path, 'r').read()


def read_lines(filename):
    """Read the lines in a given filename"""
    _path = os.path.join(os.path.dirname(__file__), filename)
    with open(_path, 'r') as f:
        required = f.read().splitlines()
    return required


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=read('README.md'),
    author=AUTHOR,
    author_email=AUTHORS_EMAIL_ADDRESS,
    url=URL,
    # license=read('LICENSE.txt'),
    keywords='Gergely Bath AI',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=read_lines('requirements.txt'),
    include_package_data=False,
    extras_require={
        'all': read_lines('requirements.txt'),  # + read_lines('requirements_dev.txt')
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
