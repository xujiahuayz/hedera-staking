from setuptools import setup, find_packages

setup(
    name="staking",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "ipykernel",
        "requests",
        "networkx",
        "matplotlib",
        "datetime",
        "openpyxl",
    ],
    extras_require={"dev": ["pylint", "black"]},
    entry_points={
        "console_scripts": ["staking=staking.cli:run"],
    },
)
