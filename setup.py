from setuptools import find_packages, setup


setup(
    name="torch_disttools",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "torch",
    ],
    extras_require={
        "tests": [
            "doit-interface",
            "flake8",
            "pytest",
            "pytest-cov",
        ],
        "docs": [
            "sphinx",
        ]
    }
)
