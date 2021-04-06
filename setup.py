#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os
from setuptools import setup

###########
# Helpers #
###########

def read_readme(fname):
    with open(
        os.path.join(os.path.dirname(__file__), fname), 
        encoding='utf-8'
    ) as f:
        return f.read()


def read_requirements(fname):
    with open(
        os.path.join(os.path.dirname(__file__), fname), 
        encoding='utf-8'
    ) as f:
        return [s.strip() for s in f.readlines()]     


setup(
    name="synergos_algorithm",
    version="0.2.0",
    author="AI Singapore",
    author_email='synergos-ext@aisingapore.org',
    description="Algorithmic component for the Synergos network",
    long_description=read_readme('README.md'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords="synergos horizontal vertical federated learning algorithm",
    url="https://gitlab.int.aisingapore.org/aims/federatedlearning/synergos_algorithm",
    license="MIT",
    packages=["synalgo"],
    python_requires = ">=3.7",
    install_requires=read_requirements("requirements.txt"),
    include_package_data=True,
    zip_safe=False
)
