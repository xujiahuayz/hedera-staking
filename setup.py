from setuptools import setup, find_packages

setup(
    name="staking",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "requests",
        "math",
        "networkx",
        "matplotlib",
        "datetime",
        "openpyxl",
    ],
    extras_require={"dev": ["pylint", "black"]},
)
