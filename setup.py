import io
from setuptools import setup, find_packages

with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='SentEval',
    version='0.1.0',
    url='https://github.com/goel96vibhor/CIFAR-10.1',
    packages=find_packages(exclude=['examples']),
    license='Attribution-NonCommercial 4.0 International',
    long_description=readme,
)

