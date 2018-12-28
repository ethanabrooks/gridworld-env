#! /usr/bin/env python

# third party
from setuptools import find_packages, setup

setup(
    name='gridworld-env',
    version='0.0.0',
    description='A customizable library of gridworlds conforming to the OpenAI Gym (https://gym.openai.com/) API'
    url='https://github.com/lobachevzky/gridworld-env',
    author='Ethan Brooks',
    author_email='ethanabrooks@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    install_requires=['gym==0.10.4', 'numpy'])
