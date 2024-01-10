from setuptools import setup, find_packages

setup(
    name="nggp_lib",  # Replace with your package's name
    version="0.1",  # Replace with your package's version
    author="Jesse Gill",  # Replace with your name
    author_email="s2045318ed.ac.uk",  # Replace with your email
    description="Adaption of NGGP ",  # Provide a short description
    long_description=open("README.md").read(),  # Long description read from the README.md
    long_description_content_type="text/markdown",
    url="http://github.com/yourusername/your_package",  # Replace with the URL to your package's repo
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        # List of dependencies, e.g.,
        # "numpy >= 1.18.1",
        # "pandas >= 1.0.3",
    ],
    classifiers=[
        # Choose appropriate classifiers from https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Minimum version requirement of the package
)
