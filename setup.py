from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="AutoDH",
    version="0.2",
    author="Eddie Groshev",
    url="https://github.com/edsterg/autodh",
    description="Library for automatic extraction of Denavit-Hartenberg parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
    keywords="autdh automatic dh denavit-hartenberg robotics robot",
    license="MIT",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests", "tests.*"]),
    install_requires=[
        "numpy",
        "prettytable",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-mock",
        ],
    },
)
