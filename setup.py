from setuptools import setup, find_packages
from os import path

working_directory = path.abspath(path.dirname(__file__))

with open(path.join(working_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="agdiff",
    version="0.0.1",
    description='PyTorch Implementation of AGDIFF from "AGDIFF: Attention-Enhanced Diffusion for Molecular Geometry Prediction"',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Fatemeh Fathi Niazi",
    author_email="fathinia@msu.edu",
    url="https://github.com/ADicksonLab/AGDIFF",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["agdiff", "diffusion models", "generative models", "conformer"],
)
