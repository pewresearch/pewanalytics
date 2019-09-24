from setuptools import setup, find_packages
from distutils.core import setup


with open("README.md") as README:
    readme = str(README.read())

with open("requirements.txt") as reqs:
    install_requires = [
        str(line)
        for line in reqs.read().split("\n")
        if line and not line.startswith(("--", "git+ssh"))
    ]

setup(
    name="pewanalytics",
    version="0.0.1",
    description="Analytics utilities for text processing.",
    long_description=readme,
    url="https://github.com/pewresearch/pewanalytics",
    author="Pew Research Center",
    author_email="admin@pewresearch.tech",
    keywords="statistics nlp text",
    license="MIT",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    include_package_data=True,
    install_requires=install_requires,
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        # 'Development Status :: 1 - Planning',
        "Development Status :: 2 - Pre-Alpha",
        # 'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',
        # 'Development Status :: 6 - Mature',
        # 'Development Status :: 7 - Inactive'
        "Environment :: Web Environment",
        "Framework :: Sphinx",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
