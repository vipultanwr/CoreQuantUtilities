from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="CoreQuantUtilities",
    version="1.2.0",
    packages=find_packages(),
    description="A core library for quantitative analysis utilities.",
    author="Vipul Tanwar",
    author_email="",
    url="https://github.com/vipultanwr/CoreQuantUtilities",  # Assuming this is your repo URL
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
