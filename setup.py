from setuptools import setup, find_packages

setup(
    name="sam3",
    version="1.0.0",
    description="SAM3 (Segment Anything Model 3) — Meta official model library",
    packages=find_packages(include=["sam3", "sam3.*"]),
    python_requires=">=3.10",
)
