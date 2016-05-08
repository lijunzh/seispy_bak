# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='seispy',
    version='0.1.0',
    description="Lijun's personal toolbox for seismic signal processing",
    long_description=readme,
    author='Lijun Zhu',
    author_email='gatechzhu@gmail.com',
    url='https://github.com/gatechzhu/seispy',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
