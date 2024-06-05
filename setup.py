import os
from setuptools import setup, find_packages

def read_requirements():
    return open(os.path.join(os.path.dirname(__file__), 'requirements.txt')).readlines()

setup(
    name='uld',
    version='1.0',
    packages=find_packages(),
    install_requires=read_requirements(),
    author='Jiabao Ji',
    author_email='jiabaoji@ucsb.edu',
    description='This is ',
    license='MIT',
    keywords='LLM Unlearning, LLM, Machine Leanring',
    url='',
)
