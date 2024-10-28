import pathlib
from pip.req import parse_requirements
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

reqs = [str(ir.req) for ir in parse_requirements(here / 'requirements.txt')]

setup(
    name='latengen',
    version='0.1.0',
    description='A framework of AI methods for predicting and refining behaviour of volcanic eruptions',
    long_description=long_description,
    url='https://github.com/FREVA-CLINT/climatereconstructionAI',
    author='Climate Informatics and Technology group at DKRZ (Deutsches Klimarechenzentrum)',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'License :: OSI Approved :: CC0 License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "Programming Language :: Python :: 3.12",
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='climate, artificial intelligence, generative AI, unstructured data',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.7, <4',
    include_package_data=True,
    install_requires=reqs,
)
