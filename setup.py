from setuptools import find_packages, setup

setup(
    name="CoreQuantUtilities",
    version="0.1.0",
    packages=find_packages(),
    description="A core library for quantitative analysis utilities.",
    author="Vipul Tanwar",
    author_email="",
    url="<your-git-repo-url>",  # Replace with your git repo URL
    install_requires=[
        # Dependencies will be read from requirements.txt
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
