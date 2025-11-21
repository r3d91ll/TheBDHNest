from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bdhnest",
    version="0.1.0",
    description="Production monitoring tools for BDH neural architecture training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Todd Bucy",
    url="https://github.com/r3d91ll/TheBDHNest",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "dash>=2.14.0",
        "dash-bootstrap-components>=1.5.0",
        "plotly>=5.17.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "fisher": ["scipy>=1.11.0", "scikit-learn>=1.3.0"],
    },
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "bdhnest=bdhnest.dashboard:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
